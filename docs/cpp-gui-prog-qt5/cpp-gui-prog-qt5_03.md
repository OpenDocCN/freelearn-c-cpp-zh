# 数据库连接

在上一章中，我们学习了如何从头开始创建一个登录页面。然而，它还没有功能，因为登录页面还没有连接到数据库。在本章中，您将学习如何将您的 Qt 应用程序连接到验证登录凭据的 MySQL（或 MariaDB）数据库。

在本章中，我们将涵盖以下主题：

+   介绍 MySQL 数据库系统

+   设置 MySQL 数据库

+   SQL 命令

+   Qt 中的数据库连接

+   功能性登录页面

我们将逐步学习本章内容，以发现 Qt 提供的强大功能，使您的应用程序可以直接连接到数据库，而无需任何额外的第三方依赖。数据库查询本身是一个庞大的主题，但我们将能够通过示例和实际方法从头开始学习最基本的命令。

Qt 支持多种不同类型的数据库系统：

+   MySQL（或 MariaDB）

+   SQLite（版本 2 和 3）

+   IBM DB2

+   Oracle

+   ODBC

+   PostgreSQL

+   Sybase Adaptive Server

其中最受欢迎的两种是 MySQL 和 SQLite。SQLite 数据库通常用于离线，并且不需要任何设置，因为它使用磁盘文件格式来存储数据。因此，在本章中，我们将学习如何设置 MySQL 数据库系统，并同时学习如何将我们的 Qt 应用程序连接到 MySQL 数据库。用于连接到 MySQL 数据库的 C++代码可以在不进行太多修改的情况下重用于连接到其他数据库系统。

# 介绍 MySQL 数据库系统

**MySQL**是一种基于关系模型的开源数据库管理系统，这是现代数据库系统用于存储各种信息的最常用方法。

与一些其他传统模型（如对象数据库系统或分层数据库系统）不同，关系模型已被证明更加用户友好，并且在其他模型之外表现出色。这就是为什么我们今天看到的大多数现代数据库系统大多使用这种方法的原因。

MySQL 最初由一家名为**MySQL AB**的瑞典公司开发，其名称是公司联合创始人的女儿*My*和**Structured Query Language**的缩写*SQL*的组合。

与 Qt 类似，MySQL 在其历史上也曾被多个不同的人拥有。最引人注目的收购发生在 2008 年，**Sun Microsystems**以 10 亿美元收购了 MySQL AB。一年后的 2009 年，**Oracle Corporation**收购了 Sun Microsystems，因此 MySQL 直到今天仍归 Oracle 所有。尽管 MySQL 多次易手，但它仍然是一款开源软件，允许用户更改代码以适应其自身目的。

由于其开源性质，还有其他从 MySQL 项目派生/分叉出来的数据库系统，如**MariaDB**、**Percona Server**等。然而，这些替代方案与 MySQL 并不完全兼容，因为它们已经修改了以适应自己的需求，因此在这些系统中有些命令可能会有所不同。

根据**Stack Overflow**在 2017 年进行的一项调查，MySQL 是 Web 开发人员中使用最广泛的数据库系统，如下图所示：

![](img/ea013fb4-46cf-44fc-ac35-1968cc90e84f.png)

调查结果表明，您在本章中学到的内容不仅可以应用于 Qt 项目，还可以应用于 Web、移动应用程序和其他类型的应用程序。

此外，MySQL 及其变体被大公司和项目组使用，如 Facebook、YouTube、Twitter、NASA、Wordpress、Drupal、Airbnb、Spotify 等。这意味着在开发过程中遇到任何技术问题时，您可以轻松获得答案。

有关 MySQL 的更多信息，请访问：

[`www.mysql.com`](https://www.mysql.com)

# 设置 MySQL 数据库

设置 MySQL 数据库有许多不同的方法。这实际上取决于您正在运行的平台类型，无论是 Windows、Linux、Mac 还是其他类型的操作系统；它还将取决于您的数据库用途——无论是用于开发和测试，还是用于大规模生产服务器。

对于大规模服务（如社交媒体），最好的方法是从源代码编译 MySQL，因为这样的项目需要大量的优化、配置，有时需要定制，以处理大量用户和流量。

但是，如果您只是进行正常使用，可以直接下载预编译的二进制文件，因为默认配置对此非常足够。您可以从官方网站或下载安装包安装独立的 MySQL 安装程序，该安装程序还包括 MySQL 以外的几个其他软件。

在本章中，我们将使用一个名为**XAMPP**的软件包，这是一个由**Apache Friends**开发的 Web 服务器堆栈软件包。该软件包包括**Apache**，**MariaDB**，**PHP**和其他可选服务，您可以在安装过程中添加。以前，MySQL 是该软件包的一部分，但从 5.5.30 和 5.6.14 版本开始，它已经被**MariaDB**替换。MariaDB 几乎与 MySQL 相同，除了涉及高级功能的命令，这些功能我们在本书中不会使用。

我们使用 XAMPP 的原因是它有一个控制面板，可以轻松启动和停止服务，而无需使用命令提示符，并且可以轻松访问配置文件，而无需自己深入安装目录。对于涉及频繁测试的应用程序开发来说，它非常快速和高效。但是，不建议在生产服务器上使用 XAMPP，因为一些安全功能已经被默认禁用。

或者，您也可以通过其他类似的软件包安装 MySQL，如**AppServ**，**AMPPS**，**LAMP**（仅限 Linux），**WAMP**（仅限 Windows），**Zend****Server**等。

现在，让我们学习如何安装 XAMPP：

1.  首先，访问他们的网站[`www.apachefriends.org`](https://www.apachefriends.org)，并点击屏幕底部的一个下载按钮，显示您当前操作系统的图标：

![](img/d5053fc7-76bd-47ad-bf75-a82f1f8758db.png)

1.  一旦您点击下载按钮，下载过程应该在几秒钟内自动开始，并且一旦完成，它应该继续安装程序。在安装过程开始之前，请确保包括 Apache 和 MySQL/MariaDB。

1.  安装 XAMPP 后，从开始菜单或桌面快捷方式启动控制面板。之后，您可能会注意到没有发生任何事情。这是因为 XAMPP 控制面板默认隐藏在任务栏中。您可以通过右键单击它并在弹出菜单中选择显示/隐藏选项来显示控制面板窗口。以下屏幕截图显示了 Windows 机器上的情况。对于 Linux，菜单可能看起来略有不同，但总体上非常相似。对于 macOS，您必须从启动台或从 dock 启动 XAMPP：

![](img/2a3b8dac-a0be-4737-9347-82c56c7345a1.png)

1.  一旦您点击显示/隐藏选项，您最终将在屏幕上看到控制面板窗口。如果再次点击显示/隐藏选项，窗口将被隐藏起来：

![](img/4984acd2-2086-41b0-bbcd-962dc8e2a416.png)

1.  他们的控制面板乍一看就很容易理解。在左侧，您可以看到 XAMPP 中可用服务的名称，在右侧，您将看到指示启动、配置、日志等按钮。由于某种原因，XAMPP 显示 MySQL 作为模块名称，但实际上它正在运行 MariaDB。不用担心；由于 MariaDB 是 MySQL 的一个分支，两者基本上工作方式相同。

1.  在本章中，我们只需要 Apache 和 MySQL（MariaDB），所以让我们点击这些服务的启动按钮。一两秒后，您会看到启动按钮现在标记为停止，这意味着服务已经启动！：

![](img/ac835466-9711-40f6-b924-8cecf413aeff.png)

1.  要验证这一点，让我们打开浏览器，输入`localhost`作为网站地址。如果您看到类似以下图像的东西，这意味着 Apache Web 服务器已成功启动！：

![](img/3b8247ee-f9db-4a17-a1a8-ccc18d7147e4.png)

1.  Apache 在这里非常重要，因为我们将使用它来使用名为**phpMyAdmin**的基于 Web 的管理工具来配置数据库。phpMyAdmin 是用 PHP 脚本语言编写的 MySQL 管理工具，因此得名。尽管它最初是为 MySQL 设计的，但它对 MariaDB 也非常有效。

1.  要访问 phpMyAdmin 控制面板，请在浏览器上输入`localhost/phpmyadmin`。之后，您应该会看到类似于这样的东西：

![](img/5b480b92-1af6-46ae-8bf8-e0a965c6b8a0.png)

1.  在页面的左侧，您将看到导航面板，它允许您访问 MariaDB 数据库中可用的不同数据库。页面的右侧是各种工具，让您查看表格，编辑表格，运行 SQL 命令，将数据导出到电子表格，设置权限等等。

1.  默认情况下，您只能在右侧的设置面板上修改数据库的常规设置。在能够修改特定数据库的设置之前，您必须在左侧的导航面板上选择一个数据库。

1.  数据库就像一个您可以在其中存储日志的文件柜。每本日志称为一个表，每个表包含数据，这些数据像电子表格一样排序。当您想从 MariaDB 获取数据时，您必须在获取数据之前指定要访问的文件柜（数据库）和日志（表）。希望这能让您更好地理解 MariaDB 和其他类似的数据库系统背后的概念。

1.  现在，让我们开始创建我们的第一个数据库！要这样做，您可以点击导航面板上方的数据库名称上方的新建按钮，或者点击菜单顶部的数据库按钮。这两个按钮都会带您到数据库页面，您应该能够在菜单按钮下方看到这个：

![](img/14b5f1e8-da81-4fc8-a2ab-32ea8391e601.png)

1.  之后，让我们创建我们的第一个数据库！输入您想要创建的数据库名称，然后点击创建按钮。数据库创建后，您将被重定向到结构页面，该页面将列出此数据库中包含的所有表。默认情况下，您新创建的数据库不包含任何表，因此您将看到一行文本，其中说没有在数据库中找到表：

![](img/e4aa4deb-6437-4a98-98f9-f8ea2582bf7c.png)

1.  猜猜我们接下来要做什么？正确，我们将创建我们的第一个表！首先，让我们插入您想要创建的表的名称。由于在本章后面我们将做一个登录页面，让我们将我们的表命名为`user`。我们将保留默认的列数，然后点击 Go。

1.  之后，您将被重定向到另一个页面，其中包含许多列的输入字段供您填写。每一列代表一个数据结构，它将在创建后添加到您的表中。

1.  第一件需要添加到表结构中的是一个 ID，它将在每次插入新数据时自动增加。然后，添加一个时间戳列来指示数据插入的日期和时间，这对于调试很有用。最后，我们将添加一个用户名列和密码列用于登录验证。如果您不确定如何操作，请参考以下图片。确保您遵循图片中被圈出的设置：

![](img/75358cae-706a-4724-bf52-f7e4b844bfff.png)

1.  结构的类型非常重要，必须根据其预期目的进行设置。例如，id 列必须设置为 INT（整数），因为它必须是一个完整的数字，而用户名和密码必须设置为 VARCHAR 或其他类似的数据类型（CHAR、TEXT 等），以便正确保存数据。

1.  另一方面，时间戳必须设置为时间戳类型，并且必须将默认值设置为 CURRENT_TIMESTAMP，这将通知 MariaDB 在数据插入时自动生成当前时间戳。

1.  请注意，ID 列的索引设置必须设置为 PRIMARY，并确保 A_I（自动增量）复选框被选中。当您选中 A_I 复选框时，将出现一个添加索引窗口。您可以保持默认设置，然后点击 Go 按钮完成步骤并开始创建表：

![](img/edd187b5-7f50-47b4-9f89-d5caee38de4c.png)

1.  创建新表后，您应该能够看到类似以下图片的内容。您仍然可以随时通过单击更改按钮来编辑结构设置；您还可以通过单击列右侧的删除按钮来删除任何列。请注意，删除列也将删除属于该列的所有现有数据，此操作无法撤消：

![](img/6342b9cf-6219-4036-92a8-e4ed95cf1a96.png)

1.  尽管我们通常会通过程序或网页向数据库添加数据，但我们也可以直接在 phpMyAdmin 上添加数据以进行测试。要使用 phpMyAdmin 添加数据，首先必须创建一个数据库和表，这是我们在前面的步骤中已经完成的。然后，点击菜单顶部的插入按钮：

![](img/bf34412c-d063-4f91-8f52-053dd8ee6768.png)

1.  之后，您会看到一个表单出现，它类似于我们之前创建的数据结构：

![](img/480a40bc-f234-468d-9184-1e3e5b45e03c.png)

1.  您可以简单地忽略 ID 和时间戳的值，因为当您保存数据时它们将自动生成。在这种情况下，只需要填写用户名和密码。为了测试，让我们将`test`作为用户名，`123456`作为密码。然后，点击 Go 按钮保存数据。

请注意，您不应该以人类可读的格式保存密码在您的实际生产服务器上。在将密码传递到数据库之前，您必须使用加密哈希函数（如 SHA-512、RIPEEMD-512、BLAKE2b 等）对密码进行加密。这将确保密码在数据库被攻破时不被黑客读取。我们将在本章末尾讨论这个话题。

现在我们已经完成了数据库的设置并插入了我们的第一个测试数据，让我们继续学习一些 SQL 命令！

# SQL 命令

大多数流行的关系数据库管理系统，如 MySQL、MariaDB、Oracle SQL、Microsoft SQL 等，都使用一种称为 SQL（结构化查询语言）的声明性语言来与数据库交互。SQL 最初是由 IBM 工程师在 20 世纪 70 年代开发的，但后来又被 Oracle Corporation 和其他当时新兴的技术公司进一步增强。

如今，SQL 已成为**美国国家标准学会**（**ANSI**）和**国际标准化组织**（**ISO**）的标准。SQL 语言自那时起已被许多不同的数据库系统采用，并成为现代时代最流行的数据库语言之一。

在本节中，我们将学习一些基本的 SQL 命令，您可以使用这些命令与您的 MariaDB 数据库进行交互，特别是用于从数据库中获取、保存、修改和删除数据。这些基本命令也可以用于其他类型的基于 SQL 的数据库系统，以及在 ANSI 和 ISO 标准下。只是，一些更高级/定制的功能在不同系统中可能有所不同，因此在使用这些高级功能之前，请确保阅读系统手册。

好的，让我们开始吧！

# SELECT

大多数 SQL 语句都是单词简短且不言自明的。例如，此语句用于从特定表中选择一个或多个列，并获取来自所述列的数据。让我们来看看一些使用`SELECT`语句的示例命令。

以下命令检索`user`表中所有列的所有数据：

```cpp
SELECT * FROM user;
```

以下命令仅从用户表中检索`username`列：

```cpp
SELECT username FROM user;
```

以下命令检索`user`表中`id`等于`1`的`username`和`password`列：

```cpp
SELECT username, password FROM user WHERE id = 1;
```

您可以使用 phpMyAdmin 自行尝试这些命令。要执行此操作，请单击 phpMyAdmin 菜单顶部的 SQL 按钮。之后，您可以在下面的文本字段中输入命令，然后单击 Go 以执行查询：

![](img/7a345815-c3c1-45bb-93aa-affbbcaee5fa.png)

要了解有关`SELECT`语句的更多信息，请参阅以下链接：

[`dev.mysql.com/doc/refman/5.7/en/select.html`](https://dev.mysql.com/doc/refman/5.7/en/select.html)

# INSERT

接下来，`INSERT`语句用于将新数据保存到数据库表中。例如：

```cpp
INSERT INTO user (username, password) VALUES ("test2", "123456");
```

上述 SQL 命令将`username`和`password`数据插入`user`表中。还有一些其他语句可以与`INSERT`一起使用，例如`LOW_PRIORITY`，`DELAYED`，`HIGH_PRIORITY`等。

请参考以下链接以了解更多关于这些选项的信息：

[`dev.mysql.com/doc/refman/5.7/en/insert.html`](https://dev.mysql.com/doc/refman/5.7/en/insert.html)

# UPDATE

`UPDATE`语句修改数据库中的现有数据。您必须为`UPDATE`命令指定条件，否则它将修改表中的每一条数据，这不是我们期望的行为。尝试以下命令，它将更改第一个用户的`username`和`password`：

```cpp
UPDATE user SET username = "test1", password = "1234321" WHERE id = 1;
```

但是，如果 ID 为`1`的用户不存在，该命令将失败。如果您提供的`username`和`password`数据与数据库中存储的数据完全匹配（没有变化），该命令还将返回状态`0 行受影响`。有关`UPDATE`语句的更多信息，请参阅以下链接：

[`dev.mysql.com/doc/refman/5.7/en/update.html`](https://dev.mysql.com/doc/refman/5.7/en/update.html)

# DELETE

`DELETE`语句从数据库的特定表中删除数据。例如，以下命令从`user`表中删除 ID 为`1`的数据：

```cpp
DELETE FROM user WHERE id = 1;
```

尽管您可以使用此语句删除不需要的数据，但不建议从数据库中删除任何数据，因为该操作无法撤消。最好在表中添加另一列，称为状态，并使用该列指示数据是否应显示。例如，如果用户在前端应用程序中删除数据，请将该数据的状态设置为（假设）`1`而不是`0`。然后，当您想要在前端显示数据时，仅显示携带`status`为`0`的数据：

![](img/7858c675-a8f2-41dd-8112-e762d1c1b0a8.png)

这样，任何意外删除的数据都可以轻松恢复。如果您只计划使用 true 或 false，也可以使用 BOOLEAN 类型。我通常使用 TINYINT，以防将来需要第三或第四状态。有关`DELETE`语句的更多信息，您可以参考以下链接：

[`dev.mysql.com/doc/refman/5.7/en/delete.html`](https://dev.mysql.com/doc/refman/5.7/en/delete.html)

# 连接

使用关系数据库管理系统的优势在于，可以轻松地将来自不同表的数据连接在一起，并以单个批量返回给用户。这极大地提高了开发人员的生产力，因为它在设计复杂的数据库结构时提供了流动性和灵活性。

MariaDB/MySQL 中有许多类型的 JOIN 语句—INNER JOIN、FULL OUTER JOIN、LEFT JOIN 和 RIGHT JOIN。这些不同的 JOIN 语句在执行时表现不同，您可以在以下图像中看到：

![](img/12b3d2f1-150c-48c3-b1d5-07f2d459d007.png)

大多数情况下，我们将使用 INNER JOIN 语句，因为它只返回两个表中具有匹配值的数据，因此只返回所需的少量数据。JOIN 命令比其他命令复杂得多，因为您需要首先设计可连接的表。在开始测试 JOIN 命令之前，让我们创建另一个表以实现这一点。我们将称这个新表为 department：

![](img/02429334-d381-4597-82cc-d533541239a6.png)

之后，添加两个部门，如下所示：

![](img/e22b95e9-823e-4da3-8983-435a77825095.png)

然后，转到用户表，在结构页面，滚动到底部，查找所示的表单，然后单击“Go”按钮：

![](img/ca934b9d-4c81-40e2-8830-f1ee803de1a0.png)

添加一个名为 deptID（代表部门 ID）的新列，并将其数据类型设置为`int`（整数）：

![](img/50b28f30-fb37-402b-9748-76115be1cd2c.png)

完成后，设置几个测试用户，并将他们的 deptID 分别设置为`1`或`2`：

![](img/fc9743b9-80f9-45b3-b8f5-3c2dc3db7af2.png)

请注意，我在这里还添加了状态列，以检查用户是否已被删除。完成后，让我们尝试运行一个示例命令！：

```cpp
SELECT my_user.username, department.name FROM (SELECT * FROM user WHERE deptID = 1) AS my_user INNER JOIN department ON department.id = my_user.deptID AND my_user.status = 0 
```

乍一看，这看起来相当复杂，但如果您将其分成几个部分，实际上并不复杂。我们将从`()`括号内的命令开始，其中我们要求 MariaDB/MySQL 选择`deptID = 1`的`user`表中的所有列：

```cpp
SELECT * FROM user WHERE deptID = 1 
```

之后，将其包含在`()`括号中，并将整个命令命名为`my_user`。之后，您可以开始使用`INNER JOIN`语句将用户表（现在称为`my_user`）与部门表进行连接。在这里，我们还添加了一些条件来查找数据，例如部门表的 ID 必须与`my_user`的`deptID`匹配，并且`my_user`的状态值必须为`0`，表示数据仍然有效，未标记为已移除：

```cpp
(SELECT * FROM user WHERE deptID = 1) AS my_user INNER JOIN department ON department.id = my_user.deptID AND my_user.status = 0 
```

最后，在前面添加以下代码以完成 SQL 命令：

```cpp
SELECT my_user.username, department.name FROM  
```

让我们尝试上述命令，看看结果是否符合您的预期。

只要表通过匹配列相互连接，您就可以使用此方法连接无限数量的表。

要了解有关**JOIN**语句的更多信息，请访问以下链接：

[`dev.mysql.com/doc/refman/5.7/en/join.html`](https://dev.mysql.com/doc/refman/5.7/en/join.html)

在本章中，我们还没有涵盖的许多其他 SQL 语句，但我们已经涵盖的基本上就是您开始所需的全部内容。

在我们进入下一部分之前，我们必须为应用程序创建一个访问 MariaDB/MySQL 数据库的用户帐户。首先，转到 phpMyAdmin 的主页，然后单击顶部菜单上的用户帐户：

![](img/e7591330-3f91-4f6c-9fe3-7cee39a65a11.png)

然后，转到底部，查找名为“添加用户帐户”的链接：

![](img/bf188b4e-ac18-4800-b4a4-f484bef7e05c.png)

一旦您进入“添加用户帐户”页面，请在登录信息表单中输入用户名和密码信息。确保主机名设置为本地：

![](img/ab7d52be-0aba-4723-b40c-a90718e74ce5.png)

然后，向下滚动并设置用户的全局权限。在数据部分启用选项就足够了，但不要启用其他选项，因为一旦您的服务器被入侵，它可能会给黑客修改数据库结构的权限。

![](img/d3803205-cff4-4c72-ae76-9da02eba99a0.png)

创建用户帐户后，请按照以下步骤允许新创建的用户访问名为 test 的数据库（或您选择的任何其他表名）：

![](img/89c0aee6-4bd7-430d-b521-4abf695e9ba9.png)

点击“Go”按钮后，您现在已经赋予了用户帐户访问数据库的权限！在下一节中，我们将学习如何将我们的 Qt 应用程序连接到数据库。

# Qt 中的数据库连接

现在我们已经学会了如何设置一个功能齐全的 MySQL/MariaDB 数据库系统，让我们再进一步，了解 Qt 中的数据库连接模块！

在我们继续处理上一章的登录页面之前，让我们首先开始一个新的 Qt 项目，这样可以更容易地演示与数据库连接相关的功能，而不会被其他东西分散注意力。这次，我们将选择名为 Qt 控制台应用程序的终端样式应用程序，因为我们不真的需要任何 GUI 来进行演示：

![](img/b62b379e-c4f9-4152-9670-799387f56f43.png)

创建新项目后，您应该只在项目中看到两个文件，即[project_name].pro 和 main.cpp：

![](img/f74bcfc6-d1f9-463d-8fc1-25c4a112d9a6.png)

您需要做的第一件事是打开您的项目文件（`.pro`），在我的情况下是 DatabaseConnection.pro，并在第一行的末尾添加`sql`关键字，如下所示：

```cpp
QT += core sql 
```

就这么简单，我们已经成功地将`sql`模块导入到了我们的 Qt 项目中！然后，打开`main.cpp`，您应该看到一个非常简单的脚本，其中只包含八行代码。这基本上是您创建一个空控制台应用程序所需的全部内容：

```cpp
#include <QCoreApplication> 
int main(int argc, char *argv[]) 
{ 
   QCoreApplication a(argc, argv); 
   return a.exec(); 
} 
```

为了连接到我们的数据库，我们必须首先将相关的头文件导入到`main.cpp`中，如下所示：

```cpp
#include <QCoreApplication> 
#include <QtSql> 
#include <QSqlDatabase> 
#include <QSqlQuery> 
#include <QDebug> 
int main(int argc, char *argv[]) 
{ 
   QCoreApplication a(argc, argv); 
   return a.exec(); 
} 
```

没有这些头文件，我们将无法使用 Qt 的`sql`模块提供的函数，这些函数是我们之前导入的。此外，我们还添加了`QDebug`头文件，以便我们可以轻松地在控制台显示上打印出任何文本（类似于 C++标准库提供的`std::cout`函数）。

接下来，我们将向`main.cpp`文件添加一些代码。在`return a.exec()`之前添加以下突出显示的代码：

```cpp
int main(int argc, char *argv[]) 
{ 
   QCoreApplication a(argc, argv); 
   QSqlDatabase db = QSqlDatabase::addDatabase("QMYSQL"); 
   db.setHostName("127.0.0.1"); 
   db.setPort(3306); 
   db.setDatabaseName("test"); 
   db.setUserName("testuser"); 
   db.setPassword("testpass"); 
   if (db.open()) 
   { 
         qDebug() << "Connected!"; 
   } 
   else 
   { 
         qDebug() << "Failed to connect."; 
         return 0; 
   } 
   return a.exec(); 
} 
```

请注意，数据库名称、用户名和密码可能与您在数据库中设置的不同，请在编译项目之前确保它们是正确的。

完成后，让我们点击“运行”按钮，看看会发生什么！：

![](img/61992a63-5c9a-4319-95a0-50e25a5ee0ad.png)

如果您看到以下错误，请不要担心：

![](img/1b176a50-7762-48d9-8b8f-a1db1165c9b8.png)

这只是因为您必须将 MariaDB Connector（或者如果您正在运行 MySQL，则是 MySQL Connector）安装到您的计算机上，并将 DLL 文件复制到 Qt 安装路径。请确保 DLL 文件与服务器的数据库库匹配。您可以打开 phpMyAdmin 的主页，查看它当前使用的库。

出于某种原因，尽管我正在运行带有 MariaDB 的 XAMPP，但这里的库名称显示为 libmysql 而不是 libmariadb，因此我不得不安装 MySQL Connector：

![](img/3b99df22-27e4-4703-92fb-9982f7e896eb.png)

如果您使用的是 MariaDB，请在以下链接下载 MariaDB Connector：

[`downloads.mariadb.org/connector-c`](https://downloads.mariadb.org/connector-c) 如果您使用的是 MySQL（或者遇到了我遇到的相同问题），请访问另一个链接并下载 MySQL 连接器：

[`dev.mysql.com/downloads/connector/cpp/`](https://dev.mysql.com/downloads/connector/cpp/)

在您下载了 MariaDB 连接器之后，请在您的计算机上安装它：

![](img/3766d4f4-d470-4bc6-8d6f-239d81751bcf.png)

上面的截图显示了 Windows 机器的安装过程。如果您使用 Linux，您必须为您的 Linux 发行版下载正确的软件包。如果您使用 Debian、Ubuntu 或其变体之一，请下载 Debian 和 Ubuntu 软件包。如果您使用 Red Hat、Fedora、CentOS 或其变体之一，请下载 Red Hat、Fedora 和 CentOS 软件包。这些软件包的安装是自动的，所以您可以放心。但是，如果您没有使用这些系统之一，您将需要下载符合您系统要求的下载页面上列出的一个 gzipped tar 文件。

有关在 Linux 上安装 MariaDB 二进制 tarballs 的更多信息，请参阅以下链接：

[`mariadb.com/kb/en/library/installing-mariadb-binary-tarballs/`](https://mariadb.com/kb/en/library/installing-mariadb-binary-tarballs/)

至于 macOS，您需要使用一个名为**Homebrew**的软件包管理器来安装 MariaDB 服务器。

有关更多信息，请查看以下链接：

[`mariadb.com/kb/en/library/installing-mariadb-on-macos-using-homebrew/`](https://mariadb.com/kb/en/library/installing-mariadb-on-macos-using-homebrew/)

安装完成后，转到其安装目录并查找 DLL 文件（MariaDB 的`libmariadb.dll`或 MySQL 的`libmysql.dll`）。对于 Linux 和 macOS，而不是 DLL，它是`libmariadb.so`或`libmysql.so`。

然后，将文件复制到应用程序的构建目录（与应用程序的可执行文件相同的文件夹）。之后，尝试再次运行您的应用程序：

![](img/b0d5c6d5-4a32-43f6-9050-27f519df0800.png)

如果您仍然收到`连接失败`的消息，但没有`QMYSQL driver not loaded`的消息，请检查您的 XAMPP 控制面板，并确保您的数据库服务正在运行；还要确保您在代码中输入的数据库名称、用户名和密码都是正确的信息。

接下来，我们可以开始尝试使用 SQL 命令！在`return a.exec()`之前添加以下代码：

```cpp
QString command = "SELECT name FROM department"; 
QSqlQuery query(db); 
if (query.exec(command)) 
{ 
   while(query.next()) 
   { 
         QString name = query.value("name").toString(); 
         qDebug() << name; 
   } 
} 
```

上述代码将命令文本发送到数据库，并同步等待来自服务器的结果返回。之后，使用`while`循环遍历每个结果并将其转换为字符串格式。然后，在控制台窗口上显示结果。如果一切顺利，您应该会看到类似这样的东西：

![](img/1de75368-744f-4875-8829-37ac18d64e11.png)

让我们尝试一些更复杂的东西：

```cpp
QString command = "SELECT my_user.username, department.name AS deptname FROM (SELECT * FROM user WHERE status = 0) AS my_user INNER JOIN department ON department.id = my_user.deptID"; 
QSqlQuery query(db); 
if (query.exec(command)) 
{ 
   while(query.next()) 
   { 
         QString username = query.value("username").toString(); 
         QString department = query.value("deptname").toString(); 
         qDebug() << username << department; 
   } 
} 
```

这一次，我们使用**INNER JOIN**来合并两个表以选择`username`和`department`名称。为了避免关于名为`name`的变量的混淆，使用`AS`语句将其重命名为`deptname`。之后，在控制台窗口上显示`username`和`department`名称：

![](img/8b7da1e9-561b-44b2-b239-72c4041f0ccd.png)

我们暂时完成了。让我们继续下一节，学习如何使我们的登录页面功能正常！

# 创建我们的功能性登录页面

既然我们已经学会了如何将我们的 Qt 应用程序连接到 MariaDB/MySQL 数据库系统，现在是时候继续在登录页面上继续工作了！在上一章中，我们学会了如何设置登录页面的 GUI。但是，它作为登录页面完全没有任何功能，因为它没有连接到数据库并验证登录凭据。因此，我们将学习如何通过赋予 Qt 的`sql`模块来实现这一点。

只是为了回顾一下——这就是登录界面的样子：

![](img/cb136c81-3dfb-4e7b-83f5-61bb09344f75.png)

现在我们需要做的第一件事是为这个登录页面中重要的小部件命名，包括用户名输入、密码输入和提交按钮。您可以通过选择小部件并在属性编辑器中查找属性来设置这些属性：

![](img/b729735b-5cfb-437d-9b48-f45a85d46314.png)

然后，将密码输入的 echoMode 设置为 Password。这个设置将通过用点替换密码来在视觉上隐藏密码：

![](img/5a5a484f-a746-4be7-8841-08707e28016e.png)

之后，右键单击提交按钮，选择转到槽... 一个窗口将弹出并询问您要使用哪个信号。选择 clicked()，然后点击确定：

![](img/aaf55004-4f8c-47b6-ba33-6aa31b5480b3.png)

一个名为`on_loginButton_clicked()`的新函数将自动添加到`MainWindow`类中。当用户按下提交按钮时，这个函数将被 Qt 触发，因此你只需要在这里编写代码来提交`username`和`password`以进行登录验证。信号和槽机制是 Qt 提供的一项特殊功能，用于对象之间的通信。当一个小部件发出信号时，另一个小部件将收到通知，并将继续运行特定的函数，该函数旨在对特定信号做出反应。

让我们来看看代码。

首先，在项目（.pro）文件中添加`sql`关键字：

`QT += core gui`

**sql**

然后，继续在`mainwindow.cpp`中添加相关的头文件：

```cpp
#ifndef MAINWINDOW_H 
#define MAINWINDOW_H 

#include <QMainWindow> 

#include <QtSql> 
#include <QSqlDatabase> 
#include <QSqlQuery> 
#include <QDebug> 
#include <QMessageBox> 
```

然后，回到`mainwindow.cpp`，在`on_loginButton_clicked()`函数中添加以下代码：

```cpp
void MainWindow::on_loginButton_clicked() 
{ 
   QString username = ui->userInput->text(); 
   QString password = ui->passwordInput->text(); 
   qDebug() << username << password; 
} 
```

现在，点击运行按钮，等待应用程序启动。然后，输入任意随机的`username`和`password`，然后点击提交按钮。您现在应该在 Qt Creator 的应用程序输出窗口中看到您的`username`和`password`被显示出来。

接下来，我们将把之前编写的 SQL 集成代码复制到`mainwindow.cpp`中：

```cpp
MainWindow::MainWindow(QWidget *parent) : 
   QMainWindow(parent), 
   ui(new Ui::MainWindow) 
{ 
   ui->setupUi(this); 

   db = QSqlDatabase::addDatabase("QMYSQL"); 
   db.setHostName("127.0.0.1"); 
   db.setPort(3306); 
   db.setDatabaseName("test"); 
   db.setUserName("testuser"); 
   db.setPassword("testpass"); 

   if (db.open()) 
   { 
         qDebug() << "Connected!"; 
   } 
   else 
   { 
         qDebug() << "Failed to connect."; 
   } 
}
```

请注意，我在数据库名称、用户名和密码中使用了一些随机文本。请确保在这里输入正确的详细信息，并确保它们与您在数据库系统中设置的内容匹配。

我们对前面的代码做了一个小改动，就是我们只需要在`mainwindow.cpp`中调用`db = QSqlDatabase::addDatabase("QMYSQL")`，而不需要类名，因为声明`QSqlDatabase db`现在已经被移到了`mainwindow.h`中：

```cpp
private: 
   Ui::MainWindow *ui; 
 QSqlDatabase db; 
```

最后，我们添加了将`username`和`password`信息与 SQL 命令结合的代码，并将整个内容发送到数据库进行执行。如果有与登录信息匹配的结果，那么意味着登录成功，否则，意味着登录失败：

```cpp
void MainWindow::on_loginButton_clicked() 
{ 
   QString username = ui->userInput->text(); 
   QString password = ui->passwordInput->text(); 

   qDebug() << username << password; 

   QString command = "SELECT * FROM user WHERE username = '" + username 
   + "' AND password = '" + password + "' AND status = 0"; 
   QSqlQuery query(db); 
   if (query.exec(command)) 
   { 
         if (query.size() > 0) 
         { 
               QMessageBox::information(this, "Login success.", "You 
               have successfully logged in!"); 
         } 
         else 
         { 
               QMessageBox::information(this, "Login failed.", "Login 
               failed. Please try again..."); 
         } 
   } 
} 
```

再次点击运行按钮，看看当您点击提交按钮时会发生什么：

![](img/a486dbd1-070a-42c9-9c64-9b6f274b0987.png)

万岁！登录页面现在已经完全可用！

# 摘要

在本章中，我们学习了如何设置数据库系统并使我们的 Qt 应用程序连接到它。在下一章中，我们将学习如何使用强大的 Qt 框架绘制图表和图表。
