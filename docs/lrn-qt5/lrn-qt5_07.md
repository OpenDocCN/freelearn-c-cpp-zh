# 持久性

在第五章中，*Data*，我们创建了一个在内存中捕获和保存数据的框架。然而，这只是故事的一半，因为如果不将数据持久化到某个外部目的地，那么一旦关闭应用程序，数据就会丢失。在本章中，我们将在之前的工作基础上，将数据保存到 SQLite 数据库中，以便它可以在应用程序的生命周期之外存在。保存后，我们还将构建用于查找、编辑和删除数据的方法。为了在各种数据模型中免费获得所有这些操作，我们将扩展我们的数据实体，以便它们可以自动加载和保存到我们的数据库，而无需我们在每个类中编写样板代码。我们将涵盖以下主题：

+   SQLite

+   主键

+   创建客户端

+   查找客户端

+   编辑客户端

+   删除客户端

# SQLite

近年来，通用数据库技术已经分化，NoSQL 和图形数据库的爆炸使得 SQL 数据库仍然非常适用，并且在许多应用程序中仍然是一个合适的选择。Qt 内置支持多种 SQL 数据库驱动程序类型，并且可以通过自定义驱动程序进行扩展。MySQL 和 PostgreSQL 是非常流行的开源 SQL 数据库引擎，并且默认情况下都受到支持，但是它们是用于服务器的，并且需要管理，这使得它们对我们的目的来说有点不必要地复杂。相反，我们将使用更轻量级的 SQLite，它通常用作客户端数据库，并且由于其小的占用空间，在移动应用程序中非常受欢迎。

根据官方网站[`www.sqlite.org`](https://www.sqlite.org/)，“SQLite 是一个独立的、高可靠性的、嵌入式的、功能齐全的、公共领域的 SQL 数据库引擎。SQLite 是世界上使用最多的数据库引擎”。配合 Qt 的 SQL 相关类，创建数据库并存储数据非常容易。

我们需要做的第一件事是将 SQL 模块添加到我们的库项目中，以便访问 Qt 的所有 SQL 功能。在`cm-lib.pro`中添加以下内容：

```cpp
QT += sql
```

接下来，我们将接受前一章讨论的内容，并在接口后面实现与数据库相关的功能。在`cm-lib/source/controllers`中创建一个新的`i-database-controller.h`头文件：

```cpp
#ifndef IDATABASECONTROLLER_H
#define IDATABASECONTROLLER_H

#include <QJsonArray>
#include <QJsonObject>
#include <QList>
#include <QObject>
#include <QString>

#include <cm-lib_global.h>

namespace cm {
namespace controllers {

class CMLIBSHARED_EXPORT IDatabaseController : public QObject
{
    Q_OBJECT

public:
    IDatabaseController(QObject* parent) : QObject(parent){}
    virtual ~IDatabaseController(){}

    virtual bool createRow(const QString& tableName, const QString& id, 
                           const QJsonObject& jsonObject) const = 0;
    virtual bool deleteRow(const QString& tableName, const QString& id) 
                                                     const = 0;
    virtual QJsonArray find(const QString& tableName, const QString& 
                                           searchText) const = 0;
    virtual QJsonObject readRow(const QString& tableName, const 
                                      QString& id) const = 0;
    virtual bool updateRow(const QString& tableName, const QString& id, 
                           const QJsonObject& jsonObject) const = 0;
};

}}

#endif
```

在这里，我们正在实现(**创建**、**读取**、**更新**和**删除**) **CRUD**的四个基本功能，这些功能与持久存储一般相关，而不仅仅是 SQL 数据库。我们还通过一个额外的`find()`方法来补充这些功能，我们将使用它来查找基于提供的搜索文本的匹配客户端数组。

现在，让我们创建一个接口的具体实现。在`cm-lib/source/controllers`中创建一个新的`DatabaseController`类。

`database-controller.h`：

```cpp
#ifndef DATABASECONTROLLER_H
#define DATABASECONTROLLER_H

#include <QObject>
#include <QScopedPointer>

#include <controllers/i-database-controller.h>

#include <cm-lib_global.h>

namespace cm {
namespace controllers {

class CMLIBSHARED_EXPORT DatabaseController : public IDatabaseController
{
    Q_OBJECT

public:
    explicit DatabaseController(QObject* parent = nullptr);
    ~DatabaseController();

    bool createRow(const QString& tableName, const QString& id, const 
                         QJsonObject& jsonObject) const override;
    bool deleteRow(const QString& tableName, const QString& id) const 
                                                            override;
    QJsonArray find(const QString& tableName, const QString& 
                                   searchText) const override;
    QJsonObject readRow(const QString& tableName, const QString& id) 
                                                  const override;
    bool updateRow(const QString& tableName, const QString& id, const 
                         QJsonObject& jsonObject) const override;

private:
    class Implementation;
    QScopedPointer<Implementation> implementation;
};

}}

#endif
```

现在，让我们逐步了解`database-controller.cpp`中的每个关键实现细节：

```cpp
class DatabaseController::Implementation
{
public:
    Implementation(DatabaseController* _databaseController)
        : databaseController(_databaseController)
    {
        if (initialise()) {
            qDebug() << "Database created using Sqlite version: " + 
                                                sqliteVersion();
            if (createTables()) {
                qDebug() << "Database tables created";
            } else {
                qDebug() << "ERROR: Unable to create database tables";
            }
        } else {
            qDebug() << "ERROR: Unable to open database";
        }
    }

    DatabaseController* databaseController{nullptr};
    QSqlDatabase database;

private:
    bool initialise()
    {
        database = QSqlDatabase::addDatabase("QSQLITE", "cm");
        database.setDatabaseName( "cm.sqlite" );
        return database.open();
    }

    bool createTables()
    {
        return createJsonTable( "client" );
    }

    bool createJsonTable(const QString& tableName) const
    {
        QSqlQuery query(database);
        QString sqlStatement = "CREATE TABLE IF NOT EXISTS " + 
         tableName + " (id text primary key, json text not null)";

        if (!query.prepare(sqlStatement)) return false;

        return query.exec();
    }

    QString sqliteVersion() const
    {
        QSqlQuery query(database);

        query.exec("SELECT sqlite_version()");

        if (query.next()) return query.value(0).toString();

        return QString::number(-1);
    }
};
```

从私有实现开始，我们将初始化分为两个操作：`initialise()`实例化一个连接到名为`cm.sqlite`的 SQLite 数据库的操作，如果数据库文件不存在，此操作将首先为我们创建数据库文件。文件将在与应用程序可执行文件相同的文件夹中创建，`createTables()`然后创建我们需要的任何表，这些表在数据库中不存在。最初，我们只需要一个名为 client 的单个表，但稍后可以轻松扩展。我们将实际创建命名表的工作委托给`createJsonTable()`方法，以便我们可以在多个表中重用它。

传统的规范化关系数据库方法是将我们的每个数据模型持久化到自己的表中，字段与类的属性匹配。回想一下第五章中的模型图，如下所示：

![](img/f194d8cd-8f52-4936-bebc-64a2f23f37a9.png)

我们可以创建一个带有“reference”和“name”字段的客户端表，一个带有“type”、“address”和其他字段的联系人表。然而，我们将利用我们已经实现的 JSON 序列化代码，并实现一个伪文档式数据库。我们将利用一个单一的客户端表，该表将存储客户端的唯一 ID 以及整个客户端对象层次结构序列化为 JSON。

最后，我们还添加了一个`sqliteVersion()`实用方法来识别数据库使用的 SQLite 版本：

```cpp
bool DatabaseController::createRow(const QString& tableName, const QString& id, const QJsonObject& jsonObject) const
{
    if (tableName.isEmpty()) return false;
    if (id.isEmpty()) return false;
    if (jsonObject.isEmpty()) return false;

    QSqlQuery query(implementation->database);

    QString sqlStatement = "INSERT OR REPLACE INTO " + tableName + " 
                            (id, json) VALUES (:id, :json)";

    if (!query.prepare(sqlStatement)) return false;

    query.bindValue(":id", QVariant(id));
    query.bindValue(":json",    
   QVariant(QJsonDocument(jsonObject).toJson(QJsonDocument::Compact)));

    if(!query.exec()) return false;

    return query.numRowsAffected() > 0;
}

bool DatabaseController::deleteRow(const QString& tableName, const QString& id) const
{
    if (tableName.isEmpty()) return false;
    if (id.isEmpty()) return false;

    QSqlQuery query(implementation->database);

    QString sqlStatement = "DELETE FROM " + tableName + " WHERE 
                            id=:id";

    if (!query.prepare(sqlStatement)) return false;

    query.bindValue(":id", QVariant(id));

    if(!query.exec()) return false;

    return query.numRowsAffected() > 0;
}

QJsonObject DatabaseController::readRow(const QString& tableName, const QString& id) const
{
    if (tableName.isEmpty()) return {};
    if (id.isEmpty()) return {};

    QSqlQuery query(implementation->database);

    QString sqlStatement = "SELECT json FROM " + tableName + " WHERE 
                            id=:id";

    if (!query.prepare(sqlStatement)) return {};

    query.bindValue(":id", QVariant(id));

    if (!query.exec()) return {};

    if (!query.first()) return {};

    auto json = query.value(0).toByteArray();
    auto jsonDocument = QJsonDocument::fromJson(json);

    if (!jsonDocument.isObject()) return {};

    return jsonDocument.object();
}

bool DatabaseController::updateRow(const QString& tableName, const QString& id, const QJsonObject& jsonObject) const
{
    if (tableName.isEmpty()) return false;
    if (id.isEmpty()) return false;
    if (jsonObject.isEmpty()) return false;

    QSqlQuery query(implementation->database);

    QString sqlStatement = "UPDATE " + tableName + " SET json=:json 
                            WHERE id=:id";

    if (!query.prepare(sqlStatement)) return false;

    query.bindValue(":id", QVariant(id));
    query.bindValue(":json", 
   QVariant(QJsonDocument(jsonObject).toJson(QJsonDocument::Compact)));

    if(!query.exec()) return false;

    return query.numRowsAffected() > 0;
}
```

CRUD 操作都是基于`QSqlQuery`类和准备的`sqlStatements`。在所有情况下，我们首先对参数进行一些例行检查，以确保我们不会做一些愚蠢的事情。然后，我们将表名连接到一个 SQL 字符串中，用`:myParameter`语法表示参数。在准备好语句之后，随后使用查询对象的`bindValue()`方法替换参数。

在创建、删除或更新行时，我们只需在查询执行时返回一个`true`/`false`的成功指示器。假设查询准备和执行没有错误，我们检查操作受影响的行数是否大于`0`。读取操作返回从匹配记录中存储的 JSON 文本解析出的 JSON 对象。如果找不到记录或无法解析 JSON，则返回默认的 JSON 对象：

```cpp
QJsonArray DatabaseController::find(const QString& tableName, const QString& searchText) const
{
    if (tableName.isEmpty()) return {};
    if (searchText.isEmpty()) return {};

    QSqlQuery query(implementation->database);

    QString sqlStatement = "SELECT json FROM " + tableName + " where 
                            lower(json) like :searchText";

    if (!query.prepare(sqlStatement)) return {};

    query.bindValue(":searchText", QVariant("%" + searchText.toLower() 
                                                             + "%"));

    if (!query.exec()) return {};

    QJsonArray returnValue;

    while ( query.next() ) {
        auto json = query.value(0).toByteArray();
        auto jsonDocument = QJsonDocument::fromJson(json);
        if (jsonDocument.isObject()) {
            returnValue.append(jsonDocument.object());
        }
    }

    return returnValue;
}
```

最后，`find()`方法本质上与 CRUD 操作相同，但编译一个 JSON 对象数组，因为可能有多个匹配项。请注意，我们在 SQL 语句中使用`like`关键字，结合`%`通配符字符，以查找包含搜索文本的任何 JSON。我们还将比较的两侧转换为小写，以使搜索有效地不区分大小写。

# 主键

在大多数这些操作中，使用 ID 参数作为我们表中的主键至关重要。为了支持使用这个新的数据库控制器持久化我们的实体，我们需要向我们的`Entity`类添加一个属性，用于唯一标识该实体的一个实例。

在`entity.cpp`中，向`Entity::Implementation`添加一个成员变量：

```cpp
QString id;
```

然后，在构造函数中初始化它：

```cpp
Implementation(Entity* _entity, IDatabaseController* _databaseController, const QString& _key)
    : entity(_entity)
    , databaseController(_databaseController)
    , key(_key)
    , id(QUuid::createUuid().toString())
{
}
```

当我们实例化一个新的`Entity`时，我们需要生成一个新的唯一 ID，并使用`createUuid()`方法使用 QUuid 类为我们生成。**通用唯一标识符**（**UUID**）本质上是一个随机生成的数字，然后我们将其转换为字符串格式“{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}”，其中“x”是一个十六进制数字。您需要`#include <QUuid>`。

接下来，为其提供一个公共访问器方法：

```cpp
const QString& Entity::id() const
{
    return implementation->id;
}
```

现在的挑战是，如果我们正在创建一个已经具有 ID 的`Entity`（例如，从数据库加载客户端），我们需要一些机制来用已知值覆盖生成的 ID 值。我们将在`update()`方法中执行此操作：

```cpp
void Entity::update(const QJsonObject& jsonObject)
{
    if (jsonObject.contains("id")) {
        implementation->id = jsonObject.value("id").toString();
    }

    …

}
```

同样，当我们将对象序列化为 JSON 时，我们也需要包含 ID：

```cpp
QJsonObject Entity::toJson() const
{
    QJsonObject returnValue;
    returnValue.insert("id", implementation->id);
    …
}
```

太好了！这为我们所有的数据模型自动生成了唯一的 ID，我们可以将其用作数据库表中的主键。然而，数据库表的一个常见用例是实际上存在一个非常适合用作主键的现有字段，例如国民保险号码、社会安全号码、帐户参考或站点 ID。如果设置了，让我们添加一个指定要用作 ID 的数据装饰器的机制，以覆盖默认的 UUID。

在我们的`Entity`类中，在`Implementation`中添加一个新的私有成员：

```cpp
class Entity::Implementation
{
    ...
    StringDecorator* primaryKey{nullptr};
    ...
}
```

您需要`#include` `StringDecorator`头文件。添加一个受保护的修改器方法来设置它：

```cpp
void Entity::setPrimaryKey(StringDecorator* primaryKey) 
{ 
    implementation->primaryKey = primaryKey; 
}
```

然后，我们可以调整我们的`id()`方法，以便在适当的情况下返回主键值，否则默认返回生成的 UUID 值：

```cpp
const QString& Entity::id() const
{
    if(implementation->primaryKey != nullptr && !implementation->primaryKey->value().isEmpty()) {
        return implementation->primaryKey->value();
    }
    return implementation->id;
}
```

然后，在`client.cpp`构造函数中，在我们实例化所有数据装饰器之后，我们可以指定我们要使用引用字段作为我们的主键：

```cpp
Client::Client(QObject* parent)
    : Entity(parent, "client")
{
    ...

    setPrimaryKey(reference);
}
```

让我们添加一些测试来验证这种行为。我们将验证如果设置了引用值，`id()`方法将返回该值，否则将返回一个松散地符合“{xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}”格式的生成的 UUID。

在`cm-tests`项目的`client-tests.h`中，在私有槽范围中添加两个新测试：

```cpp
void id_givenPrimaryKeyWithNoValue_returnsUuid();
void id_givenPrimaryKeyWithValue_returnsPrimaryKey();
```

然后，在`client-tests.cpp`中实现测试：

```cpp
void ClientTests::id_givenPrimaryKeyWithNoValue_returnsUuid()
{
    Client testClient(this);

    // Using individual character checks
    QCOMPARE(testClient.id().left(1), QString("{"));
    QCOMPARE(testClient.id().mid(9, 1), QString("-"));
    QCOMPARE(testClient.id().mid(14, 1), QString("-"));
    QCOMPARE(testClient.id().mid(19, 1), QString("-"));
    QCOMPARE(testClient.id().mid(24, 1), QString("-"));
    QCOMPARE(testClient.id().right(1), QString("}"));

    // Using regular expression pattern matching
    QVERIFY(QRegularExpression("\\{.{8}-(.{4})-(.{4})-(.{4})-(.
                        {12})\\}").match(testClient.id()).hasMatch());
}

void ClientTests::id_givenPrimaryKeyWithValue_returnsPrimaryKey()
{
    Client testClient(this, QJsonDocument::fromJson(jsonByteArray).object());
    QCOMPARE(testClient.reference->value(), QString("CM0001"));
    QCOMPARE(testClient.id(), testClient.reference->value());
}
```

请注意，在第一个测试中，检查实际上进行了两次，只是为了演示您可以采取的几种不同方法。首先，我们使用单个字符匹配（'{'，'-'和'}'）进行检查，这相当冗长，但其他开发人员很容易阅读和理解。然后，我们再次使用 Qt 的正则表达式辅助类进行检查。这要短得多，但对于不懂正则表达式语法的普通人来说更难解析。

构建并运行测试，它们应该验证我们刚刚实施的更改。

# 创建客户端

让我们利用我们的新基础设施，并连接`CreateClientView`。如果您记得，我们提供了一个保存命令，当单击时，会调用`CommandController`上的`onCreateClientSaveExecuted()`。为了能够执行任何有用的操作，`CommandController`需要能够序列化和保存客户端实例，并且需要一个`IDatabaseController`接口的实现来为我们执行创建操作。

将它们注入到`command-controller.h`中的构造函数中，包括任何必要的头文件：

```cpp
explicit CommandController(QObject* _parent = nullptr, IDatabaseController* databaseController = nullptr, models::Client* newClient = nullptr);
```

正如我们现在已经看到了几次，将成员变量添加到`Implementation`中：

```cpp
IDatabaseController* databaseController{nullptr};
Client* newClient{nullptr};
```

将它们通过`CommandController`构造函数传递到 Implementation 构造函数：

```cpp
Implementation(CommandController* _commandController, IDatabaseController* _databaseController, Client* _newClient)
    : commandController(_commandController)
    , databaseController(_databaseController)
    , newClient(_newClient)           
{
    ...
}
```

```cpp
CommandController::CommandController(QObject* parent, IDatabaseController* databaseController, Client* newClient)
    : QObject(parent)
{
    implementation.reset(new Implementation(this, databaseController, newClient));
}
```

现在我们可以更新`onCreateClientSaveExecuted()`方法来创建我们的新客户端：

```cpp
void CommandController::onCreateClientSaveExecuted()
{
    qDebug() << "You executed the Save command!";

    implementation->databaseController->createRow(implementation->newClient->key(), implementation->newClient->id(), implementation->newClient->toJson());

    qDebug() << "New client saved.";
}
```

我们的客户端实例为我们提供了保存到数据库所需的所有信息，数据库控制器执行数据库交互。

我们的`CommandController`现在已经准备就绪，但我们实际上还没有注入数据库控制器或新客户端，因此转到`master-controller.cpp`，并像我们在`CommandController`和`NavigationController`中一样添加一个`DatabaseController`实例。添加一个私有成员，访问器方法和`Q_PROPERTY`。

在`Implementation`构造函数中，我们需要确保在初始化`CommandController`之前初始化新的客户端和`DatabaseController`，然后通过指针传递：

```cpp
Implementation(MasterController* _masterController)
    : masterController(_masterController)
{
    databaseController = new DatabaseController(masterController);
    navigationController = new NavigationController(masterController);
    newClient = new Client(masterController);
    commandController = new CommandController(masterController, databaseController, newClient);
}
```

构建和运行`cm-ui`，您应该在应用程序输出中看到新实例化的`DatabaseController`的消息，告诉您它已经创建了数据库和表：

```cpp
Database created using Sqlite version: 3.20.1
Database tables created
```

查看您的二进制文件所在的输出文件夹，您将看到一个新的`cm.sqlite`文件。

如果您导航到创建客户端视图，输入名称，然后单击保存按钮，您将看到进一步的输出，确认新客户端已成功保存：

```cpp
You executed the Save command!
New client saved
```

让我们来看看我们的数据库内部，并查看为我们完成了哪些工作。有几个 SQLite 浏览应用程序和 Web 浏览器插件可用，但我倾向于使用的是[`sqlitebrowser.org/`](http://sqlitebrowser.org/)上找到的一个。下载并安装这个，或者您选择的任何其他客户端适用于您的操作系统，并打开`cm.sqlite`文件：

![](img/035414d6-2999-4408-b674-f05e382fe65d.png)

您将看到我们有一个客户端表，就像我们要求的那样，有两个字段：id 和 json。浏览客户端表的数据，您将看到我们新创建的记录，其中包含我们在 UI 上输入的名称属性：

![](img/63acbb6f-08c5-42b8-8189-e2cc4d95bb86.png)

太棒了，我们已经在数据库中创建了我们的第一个客户端。请注意，`DatabaseController`初始化方法是幂等的，因此您可以再次启动应用程序，现有的数据库不会受到影响。同样，如果您手动删除`cm.sqlite`文件，然后启动应用程序将为您创建一个新版本（不包括旧数据），这是一种简单的删除测试数据的方法。

让我们快速调整一下，添加客户的`reference`属性。在`CreateClientView`中，复制绑定到`ui_name`的`StringEditorSingleLine`组件，并将新控件绑定到`ui_reference`。构建、运行，并创建一个新的客户：

![](img/2c9bf851-c9e6-4512-9e50-8cffc2b6c766.png)

我们的新客户愉快地使用指定的客户引用作为唯一的主键：

![](img/fcbd34f0-f483-49be-beed-c0b2fb8dda95.png)

# 面板

现在，让我们稍微完善一下我们的`CreateClientView`，这样我们就可以保存一些有意义的数据，而不仅仅是一堆空字符串。我们还有很多字段要添加，所以我们会稍微分开一些东西，并且通过将它们封装在具有描述性标题和下拉阴影的离散面板中，从视觉上将数据与不同的模型分开，为我们的 UI 增添一些活力：

![](img/d1af194c-0617-4db3-ac07-4fb20013f77c.png)

我们将首先创建一个通用的面板组件。在`cm-ui/components`中创建一个名为`Panel.qml`的新的 QML 文件。更新`components.qrc`和`qmldir`，就像我们为所有其他组件所做的那样：

```cpp
import QtQuick 2.9
import assets 1.0

Item {
    implicitWidth: parent.width
    implicitHeight: headerBackground.height +    
    contentLoader.implicitHeight + (Style.sizeControlSpacing * 2)
    property alias headerText: title.text
    property alias contentComponent: contentLoader.sourceComponent

    Rectangle {
        id: shadow
        width: parent.width
        height: parent.height
        x: Style.sizeShadowOffset
        y: Style.sizeShadowOffset
        color: Style.colourShadow
    }

    Rectangle {
        id: headerBackground
        anchors {
            top: parent.top
            left: parent.left
            right: parent.right
        }
        height: Style.heightPanelHeader
        color: Style.colourPanelHeaderBackground

        Text {
            id: title
            text: "Set Me!"
            anchors {
                fill: parent
                margins: Style.heightDataControls / 4
            }
            color: Style.colourPanelHeaderFont
            font.pixelSize: Style.pixelSizePanelHeader
            verticalAlignment: Qt.AlignVCenter
        }
    }

    Rectangle {
        id: contentBackground
        anchors {
            top: headerBackground.bottom
            left: parent.left
            right: parent.right
            bottom: parent.bottom
        }
        color: Style.colourPanelBackground

        Loader {
            id: contentLoader
            anchors {
                left: parent.left
                right: parent.right
                top: parent.top
                margins: Style.sizeControlSpacing
            }
        }
    }
}
```

这是一个非常动态的组件。与我们的其他组件不同，我们在这里传递整个面板的内容，而不是传递一个字符串或者甚至是一个自定义类。我们使用`Loader`组件来实现这一点，它可以根据需要加载 QML 子树。我们别名`sourceComponent`属性，以便调用元素可以在运行时注入他们想要的内容。

由于内容的动态性，我们无法设置组件的固定大小，因此我们利用`implicitWidth`和`implicitHeight`属性告诉父元素组件希望的大小，基于标题栏的大小加上动态内容的大小。

为了渲染阴影，我们绘制一个简单的`Rectangle`，确保它首先被渲染，通过将它放在文件的顶部附近。然后我们使用`x`和`y`属性来使其与其他元素偏移，稍微向下和向下移动。然后，用于标题条和面板背景的其余`Rectangle`元素被绘制在阴影的顶部。

为了支持这里的样式，我们需要添加一系列新的`Style`属性：

```cpp
readonly property real sizeControlSpacing: 10
```

```cpp
readonly property color colourPanelBackground: "#ffffff"
readonly property color colourPanelBackgroundHover: "#ececec"
readonly property color colourPanelHeaderBackground: "#131313"
readonly property color colourPanelHeaderFont: "#ffffff"
readonly property color colourPanelFont: "#131313"
readonly property int pixelSizePanelHeader: 18
readonly property real heightPanelHeader: 40
readonly property real sizeShadowOffset: 5
readonly property color colourShadow: "#dedede"
```

接下来，让我们添加一个地址编辑组件，这样我们就可以在供应地址和账单地址上重用它。在`cm-ui/components`中创建一个名为`AddressEditor.qml`的新的 QML 文件。像之前一样更新`components.qrc`和`qmldir`。

我们将使用我们的新的`Panel`组件作为根元素，并添加一个`Address`属性，这样我们就可以传递一个任意的数据模型进行绑定：

```cpp
import QtQuick 2.9
import CM 1.0
import assets 1.0

Panel {
    property Address address

    contentComponent:
        Column {
            id: column
            spacing: Style.sizeControlSpacing
            StringEditorSingleLine {
                stringDecorator: address.ui_building
                anchors {
                    left: parent.left
                    right: parent.right
                }
            }
            StringEditorSingleLine {
                stringDecorator: address.ui_street
                anchors {
                    left: parent.left
                    right: parent.right
                }
            }
            StringEditorSingleLine {
                stringDecorator: address.ui_city
                anchors {
                    left: parent.left
                    right: parent.right
                }
            }
            StringEditorSingleLine {
                stringDecorator: address.ui_postcode
                anchors {
                    left: parent.left
                    right: parent.right
                }
            }
        }
}
```

在这里，你可以看到我们新的`Panel`组件的灵活性，这要归功于嵌入的`Loader`元素。我们可以传递任何我们想要的 QML 内容，并且它将显示在面板中。

最后，我们可以更新我们的`CreateClientView`，添加我们新重构的地址组件。我们还将客户控件移动到它们自己的面板上：

```cpp
import QtQuick 2.9
import QtQuick.Controls 2.2
import CM 1.0
import assets 1.0
import components 1.0

Item {
    property Client newClient: masterController.ui_newClient

    Column {
        spacing: Style.sizeScreenMargin
        anchors {
            left: parent.left
            right: parent.right
            top: parent.top
            margins: Style.sizeScreenMargin
        }
        Panel {
            headerText: "Client Details"
            contentComponent:
                Column {
                    spacing: Style.sizeControlSpacing
                    StringEditorSingleLine {
                        stringDecorator: newClient.ui_reference
                        anchors {
                            left: parent.left
                            right: parent.right
                        }
                    }
                    StringEditorSingleLine {
                        stringDecorator: newClient.ui_name
                        anchors {
                            left: parent.left
                            right: parent.right
                        }
                    }
                }
        }
        AddressEditor {
            address: newClient.ui_supplyAddress
            headerText: "Supply Address"
        }
        AddressEditor {
            address: newClient.ui_billingAddress
            headerText: "Billing Address"
        }
    }
    CommandBar {
        commandList: masterController.ui_commandController.ui_createClientViewContextCommands
    }
}
```

在构建和运行之前，我们只需要调整`StringEditorSingleLine`的`textLabel`的背景颜色，以使其与它们现在显示在的面板匹配：

```cpp
Rectangle {
    width: Style.widthDataControls
    height: Style.heightDataControls
    color: Style.colourPanelBackground
    Text {
        id: textLabel
        …
    }
}
```

![](img/b6e32497-89df-4bf3-ba33-42af47bc0b7b.png)

继续创建一个新的客户并检查数据库。现在你应该看到供应和账单地址的详细信息已经成功保存。我们现在已经让我们的 CRUD 操作起作用了，所以让我们继续进行‘R’。

# 查找客户

我们刚刚成功地将我们的第一个客户保存到数据库中，现在让我们看看如何找到并查看这些数据。我们将在`cm-lib`中的一个专用类中封装我们的搜索功能，所以继续在`cm-lib/source/models`中创建一个名为`ClientSearch`的新类。

`client-search.h`:

```cpp
#ifndef CLIENTSEARCH_H
#define CLIENTSEARCH_H

#include <QScopedPointer>

#include <cm-lib_global.h>
#include <controllers/i-database-controller.h>
#include <data/string-decorator.h>
#include <data/entity.h>
#include <data/entity-collection.h>
#include <models/client.h>

namespace cm {
namespace models {

class CMLIBSHARED_EXPORT ClientSearch : public data::Entity
{
    Q_OBJECT
    Q_PROPERTY( cm::data::StringDecorator* ui_searchText READ 
                                           searchText CONSTANT )
    Q_PROPERTY( QQmlListProperty<cm::models::Client> ui_searchResults 
                READ ui_searchResults NOTIFY searchResultsChanged )

public:
    ClientSearch(QObject* parent = nullptr, 
    controllers::IDatabaseController* databaseController = nullptr);
    ~ClientSearch();

    data::StringDecorator* searchText();
    QQmlListProperty<Client> ui_searchResults();
    void search();

signals:
    void searchResultsChanged();

private:
    class Implementation;
    QScopedPointer<Implementation> implementation;
};

}}

#endif
```

`client-search.cpp`:

```cpp
#include "client-search.h"
#include <QDebug>

using namespace cm::controllers;
using namespace cm::data;

namespace cm {
namespace models {

class ClientSearch::Implementation
{
public:
    Implementation(ClientSearch* _clientSearch, IDatabaseController* 
                                                _databaseController)
        : clientSearch(_clientSearch)
        , databaseController(_databaseController)
    {
    }

    ClientSearch* clientSearch{nullptr};
    IDatabaseController* databaseController{nullptr};
    data::StringDecorator* searchText{nullptr};
    data::EntityCollection<Client>* searchResults{nullptr};
};

ClientSearch::ClientSearch(QObject* parent, IDatabaseController* databaseController)
    : Entity(parent, "ClientSearch")
{
    implementation.reset(new Implementation(this, databaseController));
    implementation->searchText = static_cast<StringDecorator*>(addDataItem(new StringDecorator(this, "searchText", "Search Text")));
    implementation->searchResults = static_cast<EntityCollection<Client>*>(addChildCollection(new EntityCollection<Client>(this, "searchResults")));

    connect(implementation->searchResults, &EntityCollection<Client>::collectionChanged, this, &ClientSearch::searchResultsChanged);
}

ClientSearch::~ClientSearch()
{
}

StringDecorator* ClientSearch::searchText()
{
    return implementation->searchText;
}

QQmlListProperty<Client> ClientSearch::ui_searchResults()
{
    return QQmlListProperty<Client>(this, implementation->searchResults->derivedEntities());
}

void ClientSearch::search()
{
    qDebug() << "Searching for " << implementation->searchText->value() << "...";
}

}}
```

我们需要从用户那里捕获一些文本，使用该文本搜索数据库，并将结果显示为匹配客户的列表。我们使用`StringDecorator`来容纳文本，实现一个`search()`方法来执行搜索，最后，添加一个`EntitityCollection<Client>`来存储结果。这里还有一个额外的要点是，我们需要向 UI 发出信号，告诉它搜索结果已经改变，这样它就知道需要重新绑定列表。为此，我们使用`searchResultsChanged()`信号进行通知，并将此信号直接连接到`EntityCollection`中内置的`collectionChanged()`信号。现在，每当隐藏在`EntityCollection`中的列表更新时，UI 将自动收到更改通知，并根据需要重新绘制自己。

接下来，在`MasterController`中添加一个`ClientSearch`的实例，就像我们为新的客户模型所做的那样。添加一个名为`clientSearch`的私有成员变量，类型为`ClientSearch*`，并在`Implementation`构造函数中对其进行初始化。记得将`databaseController`依赖项传递给构造函数。现在我们正在传递越来越多的依赖项，我们需要小心初始化顺序。`ClientSearch`依赖于`DatabaseController`，当我们来实现在`CommandController`中的搜索命令时，它将依赖于`ClientSearch`。因此，请确保在初始化`ClientSearch`之前初始化`DatabaseController`，并且`CommandController`在它们两者之后初始化。完成对`MasterController`的更改后，添加一个`clientSearch()`访问器方法和一个名为`ui_clientSearch`的`Q_PROPERTY`。

和往常一样，在我们可以在 UI 中使用它之前，我们需要在 QML 子系统中注册新的类。在`main.cpp`中，`#include <models/client-search.h>`并注册新类型：

```cpp
qmlRegisterType<cm::models::ClientSearch>("CM", 1, 0, "ClientSearch");
```

有了这一切，我们可以连接我们的`FindClientView`：

```cpp
import QtQuick 2.9
import assets 1.0
import CM 1.0
import components 1.0

Item {
    property ClientSearch clientSearch: masterController.ui_clientSearch

    Rectangle {
        anchors.fill: parent
        color: Style.colourBackground

        Panel {
            id: searchPanel
            anchors {
                left: parent.left
                right: parent.right
                top: parent.top
                margins: Style.sizeScreenMargin
            }
            headerText: "Find Clients"
            contentComponent:
                StringEditorSingleLine {
                    stringDecorator: clientSearch.ui_searchText
                    anchors {
                        left: parent.left
                        right: parent.right
                    }
                }
        }
    }
}
```

我们通过`MasterController`访问`ClientSearch`实例，并使用属性创建一个快捷方式。我们还再次利用我们的新`Panel`组件，这样可以在视图之间提供一个漂亮一致的外观和感觉，而工作量很小：

![](img/ff6f1c18-de37-4492-952d-1c01d8775251.png)

下一步是添加一个命令按钮，以便我们能够发起搜索。我们在`CommandController`中完成这个操作。在我们开始命令之前，我们对`ClientSearch`实例有一个额外的依赖，所以在构造函数中添加一个参数：

```cpp
CommandController::CommandController(QObject* parent, IDatabaseController* databaseController, Client* newClient, ClientSearch* clientSearch)
    : QObject(parent)
{
    implementation.reset(new Implementation(this, databaseController, newClient, clientSearch));
}
```

像我们对`newClient`所做的那样，通过参数传递到`Implementation`类，并将其存储在一个私有成员变量中。暂时回到`MasterController`，并将`clientSearch`实例添加到`CommandController`的初始化中：

```cpp
commandController = new CommandController(masterController, databaseController, newClient, clientSearch);
```

接下来，在`CommandController`中，复制并重命名我们为创建客户视图添加的私有成员变量、访问器和`Q_PROPERTY`，这样你就会得到一个`ui_findClientViewContextCommands`属性供 UI 使用。

创建一个额外的公共槽，`onFindClientSearchExecuted()`，当我们点击搜索按钮时将被调用：

```cpp
void CommandController::onFindClientSearchExecuted()
{
    qDebug() << "You executed the Search command!";

    implementation->clientSearch->search();
}
```

现在我们为我们的查找视图有一个空的命令列表，并且有一个在点击按钮时要调用的委托；我们现在需要做的就是在`Implementation`构造函数中添加一个搜索按钮：

```cpp
Command* findClientSearchCommand = new Command( commandController, QChar( 0xf002 ), "Search" );
QObject::connect( findClientSearchCommand, &Command::executed, commandController, &CommandController::onFindClientSearchExecuted );
findClientViewContextCommands.append( findClientSearchCommand );
```

命令管道就到这里了；现在我们可以很容易地向`FindClientView`添加一个命令栏。将以下内容插入到根项目的最后一个元素中：

```cpp
CommandBar {
    commandList: masterController.ui_commandController.ui_findClientViewContextCommands
} 
```

![](img/1c44bc94-195d-43b8-b7c0-4129fb1fc62e.png)

输入一些搜索文本并点击按钮，你会看到在应用程序输出控制台中一切都按预期触发了：

```cpp
You executed the Search command!
Searching for "Testing"...
```

太好了，现在我们需要做的是获取搜索文本，查询 SQLite 数据库以获取结果列表，并在屏幕上显示这些结果。幸运的是，我们已经为查询数据库做好了准备，所以我们可以很容易地实现这一点：

```cpp
void ClientSearch::search()
{
    qDebug() << "Searching for " << implementation->searchText->value() 
                                 << "...";

    auto resultsArray = implementation->databaseController-
         >find("client", implementation->searchText->value());
    implementation->searchResults->update(resultsArray);

    qDebug() << "Found " << implementation->searchResults-
             >baseEntities().size() << " matches";
}
```

在 UI 方面还有更多工作要做来显示结果。我们需要绑定到`ui_searchResults`属性，并动态显示列表中每个客户端的某种 QML 子树。我们将使用一个新的 QML 组件`ListView`来为我们完成繁重的工作。让我们从简单开始，以演示原理，然后逐步构建。在`FindClientView`中，立即在 Panel 元素之后添加以下内容：

```cpp
ListView {
    id: itemsView
    anchors {
        top: searchPanel.bottom
        left: parent.left
        right: parent.right
        bottom: parent.bottom
        margins: Style.sizeScreenMargin
    }
    clip: true
    model: clientSearch.ui_searchResults
    delegate:
        Text {
            text: modelData.ui_reference.ui_label + ": " + 
                  modelData.ui_reference.ui_value
            font.pixelSize: Style.pixelSizeDataControls
            color: Style.colourPanelFont
        }
}
```

`ListView`的两个关键属性如下：

+   `model`，即你想要显示的项目列表

+   代理，即你想要如何在视觉上表示每个项目

在我们的情况下，我们将模型绑定到我们的`ui_searchResults`，并用一个简单的`Text`元素表示每个项目，显示客户参考编号。这里特别重要的是`modelData`属性，它被神奇地注入到代理中，为我们暴露了底层项目（在这种情况下是一个客户对象）。

构建，运行，并对你迄今为止创建的一个测试客户端的 JSON 中存在的文本进行搜索，你会发现每个结果都显示了参考编号。如果你得到了多个结果并且它们排列不正确，不要担心，因为我们无论如何都会替换代理：

![](img/a63d1111-ec63-4786-a8bc-0614cd56606f.png)

为了保持整洁，我们将编写一个新的自定义组件用作代理。在`cm-ui/components`中创建`SearchResultDelegate`，并像往常一样更新`components.qrc`和`qmldir`：

```cpp
import QtQuick 2.9
import assets 1.0
import CM 1.0

Item {
    property Client client

    implicitWidth: parent.width
    implicitHeight: Math.max(clientColumn.implicitHeight, 
    textAddress.implicitHeight) + (Style.heightDataControls / 2)

    Rectangle {
        id: background
        width: parent.width
        height: parent.height
        color: Style.colourPanelBackground

        Column {
            id: clientColumn
            width: parent / 2
            anchors {
                left: parent.left
                top: parent.top
                margins: Style.heightDataControls / 4
            }
            spacing: Style.heightDataControls / 2

            Text {
                id: textReference
                anchors.left: parent.left
                text: client.ui_reference.ui_label + ": " + 
                      client.ui_reference.ui_value
                font.pixelSize: Style.pixelSizeDataControls
                color: Style.colourPanelFont
            }
            Text {
                id: textName
                anchors.left: parent.left
                text: client.ui_name.ui_label + ": " + 
                      client.ui_name.ui_value
                font.pixelSize: Style.pixelSizeDataControls
                color: Style.colourPanelFont
            }
        }

        Text {
            id: textAddress
            anchors {
                top: parent.top
                right: parent.right
                margins: Style.heightDataControls / 4
            }
            text: client.ui_supplyAddress.ui_fullAddress
            font.pixelSize: Style.pixelSizeDataControls
            color: Style.colourPanelFont
            horizontalAlignment: Text.AlignRight
        }

        Rectangle {
            id: borderBottom
            anchors {
                bottom: parent.bottom
                left: parent.left
                right: parent.right
            }
            height: 1
            color: Style.colourPanelFont
        }

        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.PointingHandCursor
            hoverEnabled: true
            onEntered: background.state = "hover"
            onExited: background.state = ""
            onClicked: masterController.selectClient(client)
        }

        states: [
            State {
                name: "hover"
                PropertyChanges {
                    target: background
                    color: Style.colourPanelBackgroundHover
                }
            }
        ]
    }
}
```

这里并没有什么新东西，我们只是结合了其他组件中涵盖的技术。请注意，`MouseArea`元素将触发`masterController`上我们尚未实现的方法，所以如果你点击其中一个客户端时出现错误，不要担心。

在`FindClientView`中用我们的新组件替换旧的`Text`代理，使用`modelData`属性来设置`client`：

```cpp
ListView {
    id: itemsView
    ...
    delegate:
        SearchResultDelegate {
            client: modelData
        }
}
```

![](img/763dcfa3-7b48-4081-a618-625ed04edda2.png)

现在，让我们在`MasterController`上实现`selectClient()`方法：

我们可以直接从`SearchResultDelegate`发出`goEditClientView()`信号，并完全绕过`MasterController`。这是一个完全有效的方法，而且确实更简单；然而，我更喜欢通过业务逻辑层路由所有交互，即使所有业务逻辑只是发出导航信号。这意味着如果以后需要添加任何进一步的逻辑，一切都已经连接好，你不需要更改任何管道。而且，调试 C++比 QML 要容易得多。

在`master-controller.h`中，我们需要将我们的新方法添加为公共槽，因为它将直接从 UI 中调用，而 UI 无法看到常规的公共方法：

```cpp
public slots:
    void selectClient(cm::models::Client* client);
```

在`master-controller.cpp`中提供实现，简单地调用导航协调器上的相关信号，并传递客户端：

```cpp
void MasterController::selectClient(Client* client)
{
    implementation->navigationController->goEditClientView(client);
}
```

搜索和选择已经就位，现在我们可以转向编辑客户端。

# 编辑客户端

现在已经从数据库中找到并加载了现有的客户端，我们需要一种机制来查看和编辑数据。首先，让我们创建在编辑视图中将使用的上下文命令。重复我们为查找客户端视图所采取的步骤，在`CommandController`中添加一个名为`editClientViewContextCommands`的新命令列表，以及一个访问方法和`Q_PROPERTY`。

创建一个新的槽，当用户在编辑视图上保存他们的更改时调用：

```cpp
void CommandController::onEditClientSaveExecuted()
{
    qDebug() << "You executed the Save command!";
}
```

在调用时向列表添加一个新的保存命令，调用槽：

```cpp
Command* editClientSaveCommand = new Command( commandController, QChar( 0xf0c7 ), "Save" );
QObject::connect( editClientSaveCommand, &Command::executed, commandController, &CommandController::onEditClientSaveExecuted );
editClientViewContextCommands.append( editClientSaveCommand );
```

我们现在有一个可以呈现给编辑客户端视图的命令列表；然而，我们现在需要克服的一个挑战是，当我们执行这个命令时，`CommandController` 不知道它需要处理哪个客户端实例。我们不能像处理新客户端那样将选定的客户端作为依赖项传递给构造函数，因为我们不知道用户会选择哪个客户端。一个选择是将编辑命令列表从`CommandController`移出，并放入客户端模型中。然后，每个客户端实例可以向 UI 呈现自己的命令。然而，这意味着命令功能被分割，我们失去了命令控制器给我们的封装性。它还使**客户端**模型膨胀了不应该关心的功能。相反，我们将当前选定的客户端作为`CommandController`的成员添加到其中，并在用户导航到`editClientView`时设置它。在`CommandController::Implementation`中添加以下内容：

```cpp
Client* selectedClient{nullptr};
```

添加一个新的公共槽：

```cpp
void CommandController::setSelectedClient(cm::models::Client* client)
{
    implementation->selectedClient = client;
}
```

现在我们有了选定的客户端，我们可以继续完成保存槽的实现。同样，我们已经在`DatabaseController`和客户端类中完成了繁重的工作，所以这个方法非常简单：

```cpp
void CommandController::onEditClientSaveExecuted()
{
    qDebug() << "You executed the Save command!";

    implementation->databaseController->updateRow(implementation->selectedClient->key(), implementation->selectedClient->id(), implementation->selectedClient->toJson());

    qDebug() << "Updated client saved.";
}
```

从 UI 的角度来看，编辑现有客户端基本上与创建新客户端是一样的。实际上，我们甚至可能可以使用相同的视图，只是在每种情况下传入不同的客户端对象。然而，我们将保持这两个功能分开，并只是复制和调整我们已经为创建客户端编写的 QML。更新`EditClientView`：

```cpp
import QtQuick 2.9
import QtQuick.Controls 2.2
import CM 1.0
import assets 1.0
import components 1.0

Item {
    property Client selectedClient
    Component.onCompleted: masterController.ui_commandController.setSelectedClient(selectedClient)

    Rectangle {
        anchors.fill: parent
        color: Style.colourBackground
    }

    ScrollView {
        id: scrollView
        anchors {
            left: parent.left
            right: parent.right
            top: parent.top
            bottom: commandBar. top
            margins: Style.sizeScreenMargin
        }
        clip: true

        Column {
            spacing: Style.sizeScreenMargin
            width: scrollView.width

            Panel {
                headerText: "Client Details"
                contentComponent:
                    Column {
                        spacing: Style.sizeControlSpacing
                        StringEditorSingleLine {
                            stringDecorator: 
                            selectedClient.ui_reference
                            anchors {
                                left: parent.left
                                right: parent.right
                            }
                        }
                        StringEditorSingleLine {
                            stringDecorator: selectedClient.ui_name
                            anchors {
                                left: parent.left
                                right: parent.right
                            }
                        }
                    }
            }

            AddressEditor {
                address: selectedClient.ui_supplyAddress
                headerText: "Supply Address"
            }

            AddressEditor {
                address: selectedClient.ui_billingAddress
                headerText: "Billing Address"
            }
        }
    }

    CommandBar {
        id: commandBar
        commandList: masterController.ui_commandController.ui_editClientViewContextCommands
    }
}
```

我们将客户端属性更改为`MasterView`在`Connections`元素中设置的`selectedClient`属性。我们使用`Component.onCompleted`槽调用`CommandController`并设置当前选定的客户端。最后，我们更新`CommandBar`以引用我们刚刚添加的新上下文命令列表。

构建并运行，现在您应该能够对选定的客户端进行更改，并使用保存按钮更新数据库。

# 删除客户端

我们 CRUD 操作的最后一部分是删除现有客户端。让我们通过`EditClientView`上的一个新按钮触发这个操作。我们将首先向`CommandController`添加在按下按钮时将被调用的槽：

```cpp
void CommandController::onEditClientDeleteExecuted()
{
    qDebug() << "You executed the Delete command!";

    implementation->databaseController->deleteRow(implementation->selectedClient->key(), implementation->selectedClient->id());
    implementation->selectedClient = nullptr;

    qDebug() << "Client deleted.";

    implementation->clientSearch->search();
}
```

这遵循了其他槽的相同模式，只是这一次我们还清除了`selectedClient`属性，因为虽然客户端实例仍然存在于应用程序内存中，但它已经被用户语义化地删除了。我们还刷新搜索，以便从搜索结果中删除已删除的客户端。就目前而言，我们已经执行了正确的数据库交互，但用户将被留在刚刚要求删除的客户端的`editClientView`上。我们希望用户被导航回仪表板。为了做到这一点，我们需要将`NavigationController`作为`CommandController`类的附加依赖项添加进去。复制我们为`DatabaseController`依赖项所做的操作，以便我们可以将其注入到构造函数中。记得更新`MasterController`并传入导航控制器实例。

有了数据库控制器的实例，我们可以将用户发送到仪表板视图：

```cpp
void CommandController::onEditClientDeleteExecuted()
{
    ...

    implementation->navigationController->goDashboardView();
}
```

现在我们有了导航控制器，我们还可以改进创建新客户端时的体验。让用户不再停留在新客户端视图上，而是执行对新创建的客户端 ID 的搜索并将他们导航到结果。然后他们可以轻松地选择新客户端，如果他们希望查看或编辑：

```cpp
void CommandController::onCreateClientSaveExecuted()
{
    ...

    implementation->clientSearch->searchText()-
                   >setValue(implementation->newClient->id());
    implementation->clientSearch->search();
    implementation->navigationController->goFindClientView();
}
```

删除槽完成后，我们现在可以在`CommandController`的`editClientContextCommands`列表中添加一个新的删除命令：

```cpp
Command* editClientDeleteCommand = new Command( commandController, QChar( 0xf235 ), "Delete" );
QObject::connect( editClientDeleteCommand, &Command::executed, commandController, &CommandController::onEditClientDeleteExecuted );
editClientViewContextCommands.append( editClientDeleteCommand );
```

现在我们可以选择删除现有的客户端了：

![](img/a16062b1-ad68-44bf-ad11-094b3cc3c6b7.png)

如果删除客户端，您将看到该行已从数据库中删除，并且用户成功导航回仪表板。但是，您还会看到应用程序输出窗口中充满了类似`qrc:/views/EditClientView:62: TypeError: Cannot read property 'ui_billingAddress' of null`的 QML 警告。

原因是编辑视图绑定到搜索结果的客户端实例。当我们刷新搜索时，我们会删除旧的搜索结果，这意味着编辑视图现在绑定到`nullptr`，无法再访问数据。即使在刷新搜索之前导航到仪表板，也会发生这种情况，因为执行导航的信号/槽的异步性质。修复这些警告的一种方法是在视图中对所有绑定添加空检查，并在主对象为空时绑定到本地临时对象。考虑以下示例：

```cpp
StringEditorSingleLine {
    property StringDecorator temporaryObject
    stringDecorator: selectedClient ? selectedClient.ui_reference : 
    temporaryObject
    anchors {
        left: parent.left
        right: parent.right
    }
}
```

因此，如果`selectedClient`不为空，则绑定到该对象的`ui_reference`属性，否则绑定到`temporaryObject`。甚至可以在根客户端属性上添加一层间接，并替换整个客户端对象：

```cpp
property Client selectedClient
property Client localTemporaryClient
property Client clientToBindTo: selectedClient ? selectedClient : localTemporaryClient
```

在这里，`selectedClient`将像往常一样由父级设置；`localTemporaryClient`将不会被设置，因此将在本地创建一个默认实例。然后，`clientToBindTo`将选择适当的对象使用，并且所有子控件都可以绑定到该对象。由于这些绑定是动态的，如果在加载视图后删除了`selectedClient`（就像我们的情况一样），那么`clientToBindTo`将自动切换。

由于这只是一个演示项目，我们可以安全地忽略警告，因此我们在这里不会采取任何行动，以保持简单。

# 摘要

在本章中，我们为客户端模型添加了数据库持久性。我们使其通用和灵活，以便我们可以通过简单地向`DatabaseController`类添加新表来轻松持久化其他模型层次结构。我们涵盖了所有核心 CRUD 操作，包括针对整个 JSON 对象进行匹配的自由文本搜索功能。

在第八章中，*Web 请求*，我们将继续探讨超出我们应用程序范围的数据，并查看另一个极其常见的业务应用程序需求，即向 Web 服务发出 HTTP 请求。
