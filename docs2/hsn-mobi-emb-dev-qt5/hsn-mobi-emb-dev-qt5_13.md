# 使用 Qt SQL 的远程数据库

Qt SQL 不依赖于任何特定的数据库驱动程序。相同的 API 可以与各种流行的数据库后端一起使用。数据库可以拥有巨大的存储空间，而移动和嵌入式设备存储量有限，嵌入式设备比手机更有限。您将学习如何使用 Qt 通过网络远程访问数据库。

我们将在本章中介绍以下主题：

+   驱动程序

+   连接到数据库

+   创建数据库

+   向数据库添加

# 技术要求

您可以在 `cp10` 分支中获取本章的源代码，地址为 `git clone -b cp10 https://github.com/PacktPublishing/Hands-On-Mobile-and-Embedded-Development-with-Qt-5`。

您还应该为您的系统安装了 `sqlite` 或 `mysql` 软件包。

# 驱动程序是数据库后端

Qt 支持各种数据库驱动程序或后端，这些后端封装了各种系统数据库，并允许 Qt 拥有一个统一的 API 前端。Qt 支持以下数据库类型：

| 数据库类型 | 软件 |
| --- | --- |
| QDB2 | IBM Db2 |
| QIBASE | Borland InterBase |
| QMYSQL | MySQL |
| QOCI | Oracle Call Interface |
| QODBC | ODBC |
| QPSQL | PostgreSQL |
| QSQLITE | SQLite 版本 3 或以上 |
| QSQLITE2 | SQLite 版本 2 |
| QTDS | Sybase Adaptive Server |

我们将探讨 QMYSQL 类型，因为它支持远程访问。MySQL 可以安装在树莓派上。 QSQLITE3 可以在网络资源上共享并支持远程访问，iOS 和 Android 也支持 SQLite。

# 设置

MySQL 数据库需要配置以允许您远程访问它。让我们看看我们如何做到这一点：

1.  您需要安装服务器和/或客户端。

1.  然后，我们将创建数据库并在需要时使其可通过网络访问。这将通过命令行和终端应用程序来完成。

# MySQL 服务器

我使用的是 Ubuntu，因此这些命令将主要针对基于 Debian 的 Linux。如果您使用的是不同的 Linux 发行版，只有安装命令会有所不同。您应根据您的发行版安装 MySQL 服务器和客户端。创建数据库的命令将是相同的。

这是我们将如何设置服务器的方式：

1.  您需要安装 MySQL 服务器和客户端：

`sudo apt-get install mysql-server mysql-client`

1.  运行 `sudo mysql_secure_installation`，这将允许您设置 root 账户。然后，登录到 `mysql` root 账户：

`sudo mysql -u root -p`

1.  创建新的数据库 `username` 和 `password`： `GRANT ALL PRIVILEGES ON *.* TO 'username'@'localhost' IDENTIFIED BY 'password';`。将 `username` 替换为您的数据库用户，将 `password` 替换为您想要用于访问此数据库的密码。

1.  要使服务器可以从除了 localhost 之外的主机访问，请编辑 `/etc/mysql/mysql.conf.d/mysqld.cnf`。

1.  将`bind-address = localhost`行更改为`bind-address = <your ip>`，其中`<your ip>`是数据库所在机器的 IP 地址。然后，重启`mysql`服务器：

`sudo service mysql restart`

在你的 MySQL 控制台中，允许远程用户访问数据库：

`GRANT ALL ON *.* TO 'username'@'<your ip>' IDENTIFIED BY 'password';`

将`<your ip>`更改为客户端设备的 IP 地址或主机名，`username`更改为你在 MySQL 服务器上使用的用户名，`password`更改为他们将要使用的密码。

# SQLite

SQLite 是一个基于文件的数据库，因此没有服务器这样的东西。我们仍然可以通过网络文件系统远程连接到它，例如 Windows 文件共享/Samba、**网络文件系统**（NFS）或 Linux 上的**安全外壳文件系统**（SSHFS）。SSHFS 允许你像本地文件系统一样挂载和访问远程文件系统。

除非你需要，否则没有必要手动使用晦涩的命令来创建数据库，因为我们将会使用 Qt 来创建它！

在 Android 上，有 Samba 客户端，可以将 Windows 网络共享挂载，这样我们就可以使用它。如果你使用 Raspberry Pi 或其他开发板，你可能能够使用 SSHFS 通过 SSH 挂载远程目录。

# 连接到本地或远程数据库

一旦我们配置并启动了数据库，现在我们可以使用相同的函数连接到它，无论它是本地数据库还是远程数据库。现在，让我们看看如何编写代码来连接到数据库，无论是本地还是远程。

数据库要么是本地可用的，这通常意味着在同一台机器上，要么通过网络远程访问。使用 Qt 连接到这些不同的数据库基本上是相同的。并非所有数据库都支持远程访问。

首先，让我们使用本地数据库。

要使用`sql`模块，我们需要将`sql`添加到配置文件中：

`QT += sql`

要在 Qt 中连接到数据库，我们需要使用`QSqlDatabase`类。

# QSqlDatabase

尽管名字叫`QSqlDatabase`，但它代表的是对数据库的连接，而不是数据库本身。

要创建数据库连接，你首先需要指定你正在使用的数据库类型。它以支持数据库的字符串表示形式引用。让我们首先选择 MySQL 的`QMYSQL`数据库。

源代码可以在 Git 仓库的`Chapter10-1`目录下的`cp10`分支中找到。

要使用`QSqlDatabase`，我们首先需要添加数据库以创建其实例。

静态的`QSqlDatbase::addDatabase`函数接受一个参数，即数据库类型，并将数据库实例添加到数据库连接列表中。

在这里，我们添加一个 MySQL 数据库，所以使用`QMYSQL`类型：

```cpp
 QSqlDatabase db = QSqlDatabase::addDatabase("QMYSQL");
```

如果你正在连接到 SQLite 数据库，请使用`MSQLITE`数据库类型：

```cpp
QSqlDatabase db = QSqlDatabase::addDatabase("MSQLITE");
```

大多数数据库都需要用户名和密码。要设置`username`和`password`，请使用以下命令：

```cpp
db.setUserName("username");
db.setPassword("password");
```

由于我们正在连接到远程 MySQL 数据库，因此我们需要指定主机名。它也可以是一个 IP 地址：

```cpp
db.setHostName("10.0.0.243");
```

要开始连接，请调用 `open()` 函数：

```cpp
bool ok = db.open()
```

`open` 返回一个 `bool`，如果成功则为 `true`，如果失败则为 `false`，在这种情况下我们可以检查错误：

```cpp
if (!db.open()) {
    qWarning() << dq.lastError.text();
}
```

如果这成功打开，我们就连接到了数据库。

让我们实际上创建远程数据库，因为我们有需要的权限。

# 创建和打开数据库

对于 SQLite 数据库，一旦我们打开它，它就会在文件系统中创建数据库。对于 MySQL，我们必须发送 MySQL 命令来创建数据库。我们使用 `QSqlQuery` 构建 SQL 查询来完成这一点。`QSqlQuery` 将数据库对象作为参数：

```cpp
QSqlQuery query(db);
```

要发送查询，我们在 `QSqlQuery` 对象上调用 `exec()` 函数。它需要一个 `String` 作为典型的 `query` 语法：

```cpp
QString dbname = "MAEPQTdb";
if (!query.exec("CREATE DATABASE IF NOT EXISTS " + dbname)) {
    qWarning() << "Database query error" << query.lastError().text();
}
```

`dbname` 这里是任何我们希望数据库名称为 `String`；我正在使用 `MAEPQT db`。

如果这个命令失败，我们发出警告消息。如果成功，我们就继续发出命令来 `USE` 它，所以我们调用另一个 `query` 命令：

```cpp
query.exec("USE " + dbname);
```

从这里开始，我们需要创建一些表格。我会让它保持简单，并填充一些数据。

我们开始另一个查询，但使用空命令，并将 `db` 对象作为第二个参数，这将创建指定数据库上的 `QSqlQuery` 对象，但在我们准备好之前不会执行任何命令：

```cpp
QSqlQuery q("", db);
q.exec("drop table Mobile");
q.exec("drop table Embedded");
q.exec("create table Mobile (id integer primary key, Device varchar,
Model varchar, Version number)");

q.exec("create table Embedded (id integer primary key, Device varchar,Model varchar, Version number)");

```

数据库已准备就绪，因此现在我们可以添加一些数据。

# 向数据库添加数据

Qt 文档指出，保留 `QSqlDatabase` 对象不是一个最佳实践。

这里有一些不同的方法我们可以这样做：

1.  我们可以使用 `QSqlDatabase::database` 来获取已打开数据库的实例：

```cpp
QSqlDatabase db = QSqlDatabase::database("MAEPQTdb");
QSqlQuery q("", db);
q.exec("insert into Mobile values (0, 'iPhone', '6SE', '12.1.2')");
q.exec("insert into Mobile values (1, 'Moto', 'XT1710-09', '2')");
q.exec("insert into Mobile values (1, 'rpi', '1', '1')");
q.exec("insert into Mobile values (1, 'rpi', '2', '2')");
q.exec("insert into Mobile values (1, 'rpi', '3', '3')");
```

1.  我们还可以使用 `QSqlQuery` 的另一个函数，名为 `prepare()`，它使用代理变量准备查询字符串以执行。

然后，我们可以使用 `bindValue` 将值绑定到其标识符：

```cpp
q.prepare("insert into Mobile values (id,  device, model, version)"
          "values ( :id, :device, :model, :version)");

q.bindValue(":id", 0);
q.bindValue(":device", "iPhone");
q.bindValue(":model", "6SE");
q.bindValue(":version", "12.1.2");
q.exec();
```

1.  作为一种替代方案，你可以使用 `bindValue` 函数，并将第一个参数设置为标识符的位置索引，从数字 0 开始，向上通过值进行操作：

```cpp
q.bindValue(1, "iPhone");
q.bindValue(3, "12.1.2");
q.bindValue(2, "6SE");
```

1.  你也可以按值的顺序使用 `bindValue`：

```cpp
q.bindValue(0);
q.bindValue("iPhone");
q.bindValue("6SE");
q.bindValue("12.1.2");
```

接下来，让我们看看如何从数据库中检索数据。

# 执行查询

到目前为止，我们一直在运行查询，但没有返回任何数据。数据库的一个要点是查询数据，而不仅仅是输入数据。如果我们只能输入数据，那会有什么乐趣呢？Qt API 有一种方法来适应不同的语法和查询的数百万种方式。大多数情况下，它特定于需要返回的数据类型，但也特定于数据库数据本身。幸运的是，`QSqlQuery` 足够通用，查询参数是一个字符串。

# QSqlQuery

要检索数据，请使用 `QsqlQuery` 执行查询，然后使用以下函数对记录进行操作：

+   `first()`

+   `last()`

+   `next()`

+   `previous()`

+   `seek(int)`

`first()` 和 `last()` 函数分别用于检索第一条和最后一条记录。要反向遍历记录，请使用 `previous()`。`seek (int)` 函数接受一个整数作为参数，以确定要检索的记录。

我们将使用 `next()`，它将遍历查询中找到的记录：

```cpp
    QSqlDatabase db = QSqlDatabase::database("MAEPQTdb");
    QSqlQuery query("SELECT * FROM Mobile", db);
    int rowCount = 0;
    while (query.next()) {
        QString id = query.value(0).toString();
        QString device = query.value(1).toString();
        QString model = query.value(2).toString();
        QString version = query.value(3).toString();
        ui->tableWidget->setRowCount(rowCount + 1);
        ui->tableWidget->setItem(rowCount, 0, new QTableWidgetItem(id));
        ui->tableWidget->setItem(rowCount, 1, new
QTableWidgetItem(device));
        ui->tableWidget->setItem(rowCount, 2, new
QTableWidgetItem(model));
        ui->tableWidget->setItem(rowCount, 3, new
QTableWidgetItem(version));
        rowCount++;
    }
```

我们还使用 `value` 来检索每个字段的数据，它需要一个 `int`，表示从 0 开始的记录位置。

您还可以使用 `QSqlRecord` 和 `QSqlField` 来做同样的事情，但可以更清晰地了解实际发生的情况：

```cpp
QSqlField idField = record.field("id");
QSqlField deviceField = record.field("device");
QSqlField modelField = record.field("model");
QSqlField versionField = record.field("version");
qDebug() << Q_FUNC_INFO
         << modelField.name()
         << modelField.tableName()
         << modelField.value();
```

要获取记录数据，请使用 `value()`，它将返回一个 `QVariant`，表示该记录字段的值。

我们本可以使用基于模型的控件，然后使用 `QsqlQueryModel` 来执行查询。

# QSqlQueryModel

`QSqlQueryModel` 继承自 `QSqlQuery`，并返回一个可以用于基于模型的控件和其他类的模型对象。如果我们将我们的 `QTableWidget` 改为 `QTableView`，我们可以使用 `QSqlQueryModel` 作为其数据模型：

```cpp
QSqlQueryModel *model = new QSqlQueryModel;
model->setQuery("SELECT * FROM Mobile", db);
tableView->setModel(model);
```

这里是我的运行数据库示例的 Raspberry Pi（带有乐高支架！）使用 MySQL 插件远程运行：

![](img/0e6010a7-476f-4740-bd4c-a405253bbfda.jpg)

# 摘要

在本章中，我们了解到 `QSqlDatabase` 代表了对任何受支持的数据库的连接。您可以使用它来登录远程 MySQL 数据库或网络共享上的 SQLite 数据库。要执行数据检索，请使用 `QSqlQuery`。您使用相同的类来设置数据、表和其他数据库命令。如果您正在使用模型-视图应用程序，则可以使用 `QSqlDatabaseModel`。

在下一章中，我们将探讨使用 Qt Purchasing 模块进行应用内购买。
