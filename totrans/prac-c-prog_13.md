# 使用 MySQL 数据库

MySQL 是近年来最受欢迎的数据库管理系统之一。众所周知，数据库用于存储将来需要使用的数据。数据库中的数据可以通过加密来保护，并且可以建立索引以实现更快的访问。当数据量过高时，数据库管理系统比传统的顺序和随机文件处理系统更受欢迎。在数据库中存储数据是任何应用程序中的一项重要任务。

本章的重点是了解如何在数据库表中管理表行。在本章中，你将学习以下食谱：

+   显示默认 MySQL 数据库中的所有内置表

+   将信息存储到 MySQL 数据库中

+   在数据库中搜索所需信息

+   更新数据库中的信息

+   使用 C 语言从数据库中删除数据

在我们进入食谱之前，我们将回顾 MySQL 中最常用的函数。同时，确保你在实现本章中的食谱之前阅读 *附录 B* 和 *附录 C* 以安装 Cygwin 和 MySQL 服务器。

# MySQL 中的函数

在 C 编程中访问和使用 MySQL 数据库时，我们将不得不使用几个函数。让我们来看看它们。

# `mysql_init()`

这将初始化一个可以用于 `mysql_real_connect()` 方法的 `MYSQL` 对象。以下是它的语法：

```cpp
MYSQL *mysql_init(MYSQL *object)
```

如果传递的对象参数是 `NULL`，则函数初始化并返回一个新对象；否则，提供的对象将被初始化，并返回对象的地址。

# `mysql_real_connect()`

这将在指定的主机上运行的 MySQL 数据库引擎上建立连接。以下是它的语法：

```cpp
MYSQL *mysql_real_connect(MYSQL *mysqlObject, const char *hostName, const char *userid, const char *password, const char *dbase, unsigned int port, const char *socket, unsigned long flag)
```

这里：

+   `mysqlObject` 表示现有 `MYSQL` 对象的地址。

+   `hostName` 是提供主机名或 IP 地址的地方。要连接到本地主机，可以提供 `NULL` 或字符串 *localhost*。

+   `userid` 表示有效的 MySQL 登录 ID。

+   `password` 表示用户的密码。

+   `dbase` 表示需要建立连接的数据库名称。

+   `port` 是指定值 `0` 或提供 TCP/IP 连接的端口号的地方。

+   `socket` 是指定 `NULL` 或提供套接字或命名管道的地方。

+   `flag` 可以用来启用某些功能，例如处理过期的密码和在客户端/服务器协议中应用压缩，但其值通常保持为 `0`。

如果建立了连接，则函数返回 `MYSQL` 连接句柄；否则，它返回 `NULL`。

# `mysql_query()`

此函数执行提供的 SQL 查询。以下是它的语法：

```cpp
int mysql_query(MYSQL *mysqlObject, const char *sqlstmt)
```

这里：

+   `mysqlObject` 表示 `MYSQL` 对象

+   `sqlstmt` 表示包含要执行的 SQL 语句的空终止字符串

如果 SQL 语句执行成功，则函数返回 `0`；否则，它返回一个非零值。

# `mysql_use_result()`

在成功执行 SQL 语句后，此方法用于保存结果集。这意味着结果集被检索并返回。以下是其语法：

```cpp
MYSQL_RES *mysql_use_result(MYSQL *mysqlObject)
```

在此，`mysqlObject` 代表连接处理程序。

如果没有发生错误，该函数返回一个 `MYSQL_RES` 结果结构。在发生任何错误的情况下，该函数返回 `NULL`。

# mysql_fetch_row()

此函数从结果集中获取下一行。如果没有更多行在结果集中检索或发生错误，则函数返回 `NULL`。以下是其语法：

```cpp
MYSQL_ROW mysql_fetch_row(MYSQL_RES *resultset)
```

在这里，`resultset` 参数是从中获取下一行数据的集合。您可以通过使用下标 `row[0]`、`row[1]` 等来访问行的列中的值，其中 `row[0]` 表示第一列中的数据，`row[1]` 表示第二列中的数据，依此类推。

# mysql_num_fields()

这返回值的数量；即提供的行中的列。以下是其语法：

```cpp
unsigned int mysql_num_fields(MYSQL_ROW row)
```

在这里，参数行代表从 `resultset` 访问的单独行。

# mysql_free_result()

这释放了分配给结果集的内存。以下是其语法：

```cpp
void mysql_free_result(MYSQL_RES *resultset)
```

在这里，`resultset` 代表我们想要释放内存的集合。

# mysql_close()

此函数关闭先前打开的 MySQL 连接。以下是其语法：

```cpp
void mysql_close(MYSQL *mysqlObject)
```

它释放由 `mysqlObject` 参数表示的连接处理程序。该函数不返回任何值。

这涵盖了我们需要了解的用于在食谱中使用 MySQL 数据库的函数。从第二个食谱开始，我们将在一个数据库表中工作。所以，让我们开始创建一个名为 `ecommerce` 的数据库和其中的表。

# 创建 MySQL 数据库和表

打开 Cygwin 终端并使用以下命令打开 MySQL 命令行。通过此命令，我们希望通过用户 ID root 打开 MySQL，并尝试连接到运行在本地的 MySQL 服务器（`127.0.0.1`）：

```cpp
$ mysql -u root -p -h 127.0.0.1 
Enter password: 
Welcome to the MariaDB monitor.  Commands end with ; or \g.
Your MySQL connection id is 12
Server version: 5.7.14-log MySQL Community Server (GPL)
Copyright (c) 2000, 2017, Oracle, MariaDB Corporation Ab and others.
Type 'help;' or '\h' for help. Type '\c' to clear the current input statement. 
MySQL [(none)]>                                                    
```

出现的前一个 MySQL 提示确认了 `userid` 和 `password` 已正确输入，并且您已成功连接到正在运行的 MySQL 服务器。现在，我们可以继续运行 SQL 命令。

# 创建数据库

`create database` 语句创建具有指定名称的数据库。以下是其语法：

```cpp
Create database database_name;
```

在这里，`database_name` 是要创建的新数据库的名称。

让我们创建一个名为 `ecommerce` 的数据库来存储我们的食谱：

```cpp
MySQL [(none)]> create database ecommerce; 
Query OK, 1 row affected (0.01 sec)                                            
```

为了确认我们的 `ecommerce` 数据库已成功创建，我们将使用 `show databases` 语句查看 MySQL 服务器上现有的数据库列表：

```cpp
MySQL [(none)]> show databases; 
+--------------------+
| Database           | 
+--------------------+
| information_schema | 
| ecommerce          | 
| mysql              | 
| performance_schema | 
| sakila             |
| sys                | 
| world              |
+--------------------+
8 rows in set (0.00 sec)                         
```

在前面的数据库列表中，我们可以看到名称 `ecommerce`，这证实了我们的数据库已成功创建。现在，我们将应用 `use` 语句来访问 `ecommerce` 数据库，如下所示：

```cpp
MySQL [(none)]> use ecommerce;
Database changed        
```

现在，`ecommerce`数据库正在使用中，因此我们将给出的任何 SQL 命令都仅应用于`ecommerce`数据库。接下来，我们需要在我们的`ecommerce`数据库中创建一个表。用于创建数据库表的命令是`Create table`。让我们接下来讨论它。

# 创建表

这将创建一个具有指定名称的数据库表。以下是其语法：

```cpp
CREATE TABLE table_name (column_name column_type,column_name column_type,.....);
```

在这里：

+   `table_name`代表我们想要创建的表的名称。

+   `column_name`代表我们希望在表中出现的列名。

+   `column_type`代表列的数据类型。根据我们想要存储在列中的数据类型，`column_type`可以是`int`、`varchar`、`date`、`text`等等。

`create table`语句创建了一个具有三个列的`users`表：`email_address`、`password`和`address_of_delivery`。假设这个表将包含已在线下订单的用户的信息，我们将存储他们的电子邮件地址、密码以及订单需要送达的位置：

```cpp
MySQL [ecommerce]> create table users(email_address varchar(30), password varchar(30), address_of_delivery text);
Query OK, 0 rows affected (0.38 sec)                                           
```

为了确认表已成功创建，我们将使用`show tables`命令显示当前打开数据库中现有表列表，如下所示：

```cpp
MySQL [ecommerce]> show tables;
+---------------------+ 
| Tables_in_ecommerce | 
+---------------------+ 
| users               | 
+---------------------+ 
1 row in set (0.00 sec)         
```

`show tables`命令的输出显示了`users`表，从而确认表确实已成功创建。为了查看表结构（即其列名、列类型和列宽度），我们将使用`describe`语句。以下语句显示了`users`表的结构：

```cpp
MySQL [ecommerce]> describe users;
+---------------------+-------------+------+-----+---------+-------+
| Field | Type | Null | Key | Default | Extra |
+---------------------+-------------+------+-----+---------+-------+
| email_address | varchar(30) | YES | | NULL | |
| password | varchar(30) | YES | | NULL | |
| address_of_delivery | text | YES | | NULL | |
+---------------------+-------------+------+-----+---------+-------+
3 rows in set (0.04 sec)  
```

因此，现在我们已经了解了与数据库交互的一些基本命令，我们可以开始本章的第一个教程。

# 显示默认 mysql 数据库中的所有内置表

MySQL 服务器在安装时附带了一些默认数据库。其中之一是`mysql`。在本教程中，我们将学习如何显示`mysql`数据库中所有可用的表名。

# 如何操作...

1.  创建一个 MySQL 对象：

```cpp
mysql_init(NULL);
```

1.  建立与指定主机上运行的 MySQL 服务器的连接。同时连接到所需的数据库：

```cpp
mysql_real_connect(conn, server, user, password, database, 0, NULL, 0)
```

1.  创建一个包含`show tables`的执行 SQL 语句：

```cpp
mysql_query(conn, "show tables")
```

1.  将执行 SQL 查询的结果（即`mysql`数据库的表信息）保存到`resultset`中：

```cpp
res = mysql_use_result(conn);
```

1.  从`resultset`中逐行获取数据，并在`while`循环中仅显示该行的表名：

```cpp
while ((row = mysql_fetch_row(res)) != NULL)
     printf("%s \n", row[0]);
```

1.  释放分配给`resultset`的内存：

```cpp
mysql_free_result(res);
```

1.  关闭打开的连接处理器：

```cpp
mysql_close(conn);
```

显示内置`mysql`数据库中所有表的`mysql1.c`程序如下：

```cpp
#include <mysql/mysql.h>
#include <stdio.h>
#include <stdlib.h>

void main() {
     MYSQL *conn;
     MYSQL_RES *res;
     MYSQL_ROW row;
     char *server = "127.0.0.1";
     char *user = "root";
     char *password = "Bintu2018$";
     char *database = "mysql";
     conn = mysql_init(NULL);
     if (!mysql_real_connect(conn, server,
         user, password, database, 0, NULL, 0)) {
        fprintf(stderr, "%s\n", mysql_error(conn));
        exit(1);
    }
    if (mysql_query(conn, "show tables")) {
        fprintf(stderr, "%s\n", mysql_error(conn));
        exit(1);
    }
    res = mysql_use_result(conn);
    printf("MySQL Tables in mysql database:\n");
    while ((row = mysql_fetch_row(res)) != NULL)
        printf("%s \n", row[0]);
    mysql_free_result(res);
    mysql_close(conn);
}
```

现在，让我们深入幕后，更好地理解代码。

# 它是如何工作的...

我们将首先与 MySQL 服务器建立连接，为此，我们需要调用`mysql_real_connect`函数。但是，我们必须将一个`MYSQL`对象传递给`mysql_real_connect`函数，并且必须调用`mysql_init`函数来创建`MYSQL`对象。因此，首先调用`mysql_init`函数来初始化一个名为`conn`的`MYSQL`对象。

然后，我们将`MYSQL`对象`conn`以及有效的用户 ID、密码和主机详情一起提供给`mysql_real_connect`函数。`mysql_real_connect`函数将建立与指定主机上运行的 MySQL 服务器的连接。除此之外，该函数还将链接到提供的`mysql`数据库，并将`conn`声明为连接处理器。这意味着`conn`将在整个程序中用于执行对指定 MySQL 服务器和`mysql`数据库的任何操作。

如果在建立与 MySQL 数据库引擎的连接过程中发生任何错误，程序将在显示错误消息后终止。如果成功建立了与 MySQL 数据库引擎的连接，将调用`mysql_query`函数，并将 SQL 语句`show tables`和连接处理器`conn`提供给它。`mysql_query`函数将执行提供的 SQL 语句。为了保存`mysql`数据库的结果表信息，将调用`mysql_use_result`函数。从`mysql_use_result`函数接收到的表信息将被分配给`resultset` `res`。

接下来，我们将在一个`while`循环中调用`mysql_fetch_row`函数，每次从`resultset` `res`中提取一行；也就是说，每次从`resultset`中提取一个表详情，并分配给数组`row`。数组`row`将包含一次一个表的完整信息。存储在`row[0]`索引中的表名将在屏幕上显示。随着`while`循环的每次迭代，下一块表信息将从`resultset` `res`中提取出来，并分配给数组`row`。因此，`mysql`数据库中的所有表名都将显示在屏幕上。

然后，我们将调用`mysql_free_result`函数来释放分配给`resultset` `res`的内存，最后，我们将调用`mysql_close`函数来关闭打开的连接处理器`conn`。

让我们使用 GCC 编译`mysql1.c`程序，如下所示：

```cpp
$ gcc mysql1.c -o mysql1 -I/usr/local/include/mysql -L/usr/local/lib/mysql -lmysqlclient          
```

如果你没有收到任何错误或警告，这意味着`mysql1.c`程序已编译成可执行文件，`mysql1.exe`。让我们运行这个可执行文件：

```cpp
$ ./mysql1 
MySQL Tables in mysql database:                                                                         columns_priv                                                                      db 
engine_cost                                                                       event
func
general_log
gtid_executed
help_category
help_keyword 
help_relation 
help_topic
innodb_index_stats
innodb_table_stats
ndb_binlog_index
plugin
proc
procs_priv
proxies_priv
server_cost
servers
slave_master_info
slave_relay_log_info
slave_worker_info
slow_log
tables_priv
time_zone
time_zone_leap_second
time_zone_name
time_zone_transition 
time_zone_transition_type 
user 
```

哇！正如你所见，输出显示了`mysql`数据库中内置表列表。现在，让我们继续到下一个菜谱！

# 在 MySQL 数据库中存储信息

在本食谱中，我们将学习如何将新行插入到`users`表中。回想一下，在本章开头，我们创建了一个名为`ecommerce`的数据库，并在该数据库中创建了一个名为`users`的表，该表具有以下列：

```cpp
email_address varchar(30)
password varchar(30) 
address_of_delivery text  
```

我们现在将向此`users`表中插入行。

# 如何做到这一点…

1.  初始化一个 MYSQL 对象：

```cpp
conn = mysql_init(NULL);
```

1.  建立与运行在本地主机的 MySQL 服务器的连接。同时，连接到您想要工作的数据库：

```cpp
mysql_real_connect(conn, server, user, password, database, 0, NULL, 0)
```

1.  输入您要将新行插入到`ecommerce`数据库中`users`表中的信息，这将包括新用户的电子邮件地址、密码和送货地址：

```cpp
printf("Enter email address: ");
scanf("%s", emailaddress);
printf("Enter password: ");
scanf("%s", upassword);
printf("Enter address of delivery: ");
getchar();
gets(deliveryaddress);
```

1.  准备一个包含此信息的 SQL `INSERT`语句；即新用户的电子邮件地址、密码和送货地址：

```cpp
strcpy(sqlquery,"INSERT INTO users(email_address, password, address_of_delivery)VALUES (\'");
strcat(sqlquery,emailaddress);
strcat(sqlquery,"\', \'");
strcat(sqlquery,upassword);
strcat(sqlquery,"\', \'");
strcat(sqlquery,deliveryaddress);
strcat(sqlquery,"\')");
```

1.  执行 SQL `INSERT`语句以将新行插入到`ecommerce`数据库中的`users`表中：

```cpp
mysql_query(conn, sqlquery)
```

1.  关闭连接处理程序：

```cpp
mysql_close(conn);
```

在以下代码中显示了用于将行插入 MySQL 数据库表的`adduser.c`程序：

```cpp
#include <mysql/mysql.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void main() {
    MYSQL *conn;
    char *server = "127.0.0.1";
    char *user = "root";
    char *password = "Bintu2018$";
    char *database = "ecommerce";
    char emailaddress[30], 
    upassword[30],deliveryaddress[255],sqlquery[255];
    conn = mysql_init(NULL);
    if (!mysql_real_connect(conn, server, user, password, database, 0, 
    NULL, 0)) {
        fprintf(stderr, "%s\n", mysql_error(conn));
        exit(1);
    }
    printf("Enter email address: ");
    scanf("%s", emailaddress);
    printf("Enter password: ");
    scanf("%s", upassword);
    printf("Enter address of delivery: ");
    getchar();
    gets(deliveryaddress);
    strcpy(sqlquery,"INSERT INTO users(email_address, password, 
    address_of_delivery)VALUES (\'");
    strcat(sqlquery,emailaddress);
    strcat(sqlquery,"\', \'");
    strcat(sqlquery,upassword);
    strcat(sqlquery,"\', \'");
    strcat(sqlquery,deliveryaddress);
    strcat(sqlquery,"\')");
    if (mysql_query(conn, sqlquery) != 0)               
    { 
        fprintf(stderr, "Row could not be inserted into users
    table\n");
        exit(1);
    } 
    printf("Row is inserted successfully in users table\n");
    mysql_close(conn);
}
```

现在，让我们深入了解代码以更好地理解它。

# 它是如何工作的...

我们首先调用`mysql_init`函数，通过名称`conn`初始化一个`MYSQL`对象。初始化后的`MYSQL`对象`conn`随后被用于调用`mysql_real_connect`函数，同时提供有效的用户 ID 和密码，这将建立与运行在本地主机的 MySQL 服务器的连接。此外，该函数还将链接到我们的`ecommerce`数据库。

如果在建立与 MySQL 数据库引擎的连接时发生任何错误，将显示错误消息，程序将终止。如果成功建立与 MySQL 数据库引擎的连接，则`conn`将作为程序其余部分的连接处理程序。

您将被提示输入要将新行插入到`ecommerce`数据库中`users`表中的信息。您将被提示输入新行信息：电子邮件地址、密码和送货地址。我们将创建一个包含此信息（电子邮件地址、密码和送货地址）的 SQL `INSERT`语句，该语句应由用户输入。之后，我们将调用`mysql_query`函数，并将 MySQL 对象`conn`和 SQL `INSERT`语句传递给它以执行 SQL 语句并将新行插入到`users`表中。

如果在执行`mysql_query`函数时发生任何错误，屏幕上将显示错误消息，程序将终止。如果新行成功插入到`users`表中，屏幕上将显示消息`Row is inserted successfully in users table`。最后，我们将调用`mysql_close`函数，并将连接处理程序`conn`传递给它以关闭连接处理程序。

让我们打开 Cygwin 终端。我们需要两个终端窗口；在一个窗口中，我们将运行 SQL 命令，在另一个窗口中，我们将编译和运行 C 语言。通过按 *Alt+F2* 打开另一个终端窗口。在第一个终端窗口中，使用以下命令调用 MySQL 命令行：

```cpp
$ mysql -u root -p -h 127.0.0.1
Enter password:
Welcome to the MariaDB monitor.  Commands end with ; or \g. 
Your MySQL connection id is 27 
Server version: 5.7.14-log MySQL Community Server (GPL) 
Copyright (c) 2000, 2017, Oracle, MariaDB Corporation Ab and others. 
Type 'help;' or '\h' for help. Type '\c' to clear the current input statement. 
```

要使用我们的 `ecommerce` 数据库，我们需要将其设置为当前数据库。因此，使用以下命令打开 `ecommerce` 数据库：

```cpp
MySQL [(none)]> use ecommerce;
Reading table information for completion of table and column names
You can turn off this feature to get a quicker startup with -A 
Database changed          
```

现在，`ecommerce` 是我们的当前数据库；也就是说，我们将执行的任何 SQL 命令都只应用于 `ecommerce` 数据库。让我们使用以下 SQL `SELECT` 命令来查看 `users` 数据库表中的现有行：

```cpp
MySQL [ecommerce]> select * from users;
Empty set (0.00 sec)  
```

给定的输出确认 `users` 表目前为空。要编译 C 程序，切换到第二个终端窗口。让我们使用 GCC 编译 `adduser.c` 程序，如下所示：

```cpp
$ gcc adduser.c -o adduser -I/usr/local/include/mysql -L/usr/local/lib/mysql -lmysqlclient        
```

如果没有错误或警告，这意味着 `adduser.c` 程序已编译成可执行文件 `adduser.exe`。让我们运行这个可执行文件：

```cpp
$./adduser 
Enter email address: bmharwani@yahoo.com 
Enter password: gold 
Enter address of delivery: 11 Hill View Street, New York, USA
Row is inserted successfully in users table 
```

给定的 C 程序输出确认新行已成功添加到 `users` 数据库表中。要确认这一点，切换到打开 MySQL 命令行的终端窗口，并使用以下命令：

```cpp
MySQL [ecommerce]> select * from users;
+---------------------+----------+------------------------------------+
| email_address       | password | address_of_delivery                |
+---------------------+----------+------------------------------------+
| bmharwani@yahoo.com | gold     | 11 Hill View Street, New York, USA | 
+---------------------+----------+------------------------------------+ 
1 row in set (0.00 sec)   
```

*Voila*！给定的输出确认通过 C 语言输入的新行已成功插入到 `users` 数据库表中。

现在，让我们继续下一个菜谱！

# 在数据库中搜索所需信息

在这个菜谱中，我们将学习如何在数据库表中搜索信息。再次强调，我们假设已经存在一个包含三个列的 `users` 表，分别是 `email_address`、`password` 和 `address_of_delivery`（请参阅本章的 *创建 MySQL 数据库和表* 部分，其中我们创建了一个 `ecommerce` 数据库并在其中创建了一个 `users` 表）。输入电子邮件地址后，菜谱将搜索整个 `users` 数据库表，如果找到与提供的电子邮件地址匹配的行，则将在屏幕上显示该用户的密码和送货地址。

# 如何做到这一点...

1.  初始化一个 MYSQL 对象：

```cpp
mysql_init(NULL);
```

1.  建立与指定主机上运行的 MySQL 服务器的连接。同时，建立与 `ecommerce` 数据库的连接：

```cpp
mysql_real_connect(conn, server, user, password, database, 0, NULL, 0)
```

1.  输入您要搜索详情的用户电子邮件地址：

```cpp
printf("Enter email address to search: ");
scanf("%s", emailaddress);
```

1.  创建一个 SQL `SELECT` 语句，搜索 `users` 表中与用户输入的电子邮件地址匹配的行：

```cpp
strcpy(sqlquery,"SELECT * FROM users where email_address like \'");
strcat(sqlquery,emailaddress);
strcat(sqlquery,"\'");
```

1.  执行 SQL `SELECT` 语句。如果 SQL 查询未执行或发生错误，则终止程序：

```cpp
if (mysql_query(conn, sqlquery) != 0)                                 
{                                                                                                                                fprintf(stderr, "No row found in the users table with this email     address\n");                                                             
    exit(1);                                                                                     }  
```

1.  如果 SQL 查询执行成功，则与指定电子邮件地址匹配的行将被检索并分配给 `resultset`：

```cpp
resultset = mysql_use_result(conn);
```

1.  使用 `while` 循环逐行从 `resultset` 中提取并分配给数组 `row`：

```cpp
while ((row = mysql_fetch_row(resultset)) != NULL)
```

1.  通过显示子脚本来显示整行信息 `row[0]`、`row[1]` 和 `row[2]`，分别：

```cpp
printf("Email Address: %s \n", row[0]);
printf("Password: %s \n", row[1]);
printf("Address of delivery: %s \n", row[2]);
```

1.  分配给`resultset`的内存将被释放：

```cpp
mysql_free_result(resultset);
```

1.  打开的连接处理器被关闭：

```cpp
mysql_close(conn);
```

在以下代码中展示了用于在 MySQL 数据库表中搜索特定行的`searchuser.c`程序：

```cpp
#include <mysql/mysql.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void main() {
    MYSQL *conn;
    MYSQL_RES *resultset;
    MYSQL_ROW row;
    char *server = "127.0.0.1";
    char *user = "root";
    char *password = "Bintu2018$";
    char *database = "ecommerce";
    char emailaddress[30], sqlquery[255];
    conn = mysql_init(NULL);
    if (!mysql_real_connect(conn, server, user, password, database, 0, 
    NULL, 0)) {
        fprintf(stderr, "%s\n", mysql_error(conn));
        exit(1);
    }
    printf("Enter email address to search: ");
    scanf("%s", emailaddress);
    strcpy(sqlquery,"SELECT * FROM users where email_address like \'");
    strcat(sqlquery,emailaddress);
    strcat(sqlquery,"\'");
    if (mysql_query(conn, sqlquery) != 0)                 
    {                  
        fprintf(stderr, "No row found in the users table with this 
    email address\n");                  
        exit(1);                                                                     
    }  
    printf("The details of the user with this email address are as 
    follows:\n");
    resultset = mysql_use_result(conn);
    while ((row = mysql_fetch_row(resultset)) != NULL)
    {
        printf("Email Address: %s \n", row[0]);
        printf("Password: %s \n", row[1]);
        printf("Address of delivery: %s \n", row[2]);
    }
    mysql_free_result(resultset);
    mysql_close(conn);
}
```

现在，让我们幕后了解一下代码，以便更好地理解它。

# 它是如何工作的...

我们将首先调用`mysql_init`函数，通过名称`conn`初始化一个`MYSQL`对象。之后，我们将调用`mysql_real_connect`函数，并将有效的用户 ID、密码和主机详情传递给该函数。`mysql_real_connect`函数将连接到在指定主机上运行的 MySQL 服务器，并将连接到提供的数据库`ecommerce`。`MYSQL`对象`conn`将作为程序其余部分的连接处理器。无论何时需要连接到 MySQL 服务器和`ecommerce`数据库，引用`conn`就足够了。

如果在建立与 MySQL 数据库引擎或`ecommerce`数据库的连接时发生任何错误，将显示错误消息，程序将终止。如果成功建立与 MySQL 数据库引擎的连接，你将被提示输入你想要搜索的用户详情的电子邮件地址。

我们将创建一个 SQL `SELECT`语句，该语句将搜索与用户输入的电子邮件地址匹配的`users`表中的行。然后，我们将调用`mysql_query`函数，并将创建的 SQL `SELECT`语句及其连接处理器`conn`传递给它。如果 SQL 查询没有执行或发生某些错误，程序将在显示错误消息后终止。如果查询成功，则通过调用`mysql_use_result`函数检索满足条件的结果行（即与提供的电子邮件地址匹配的行），并将它们分配给结果集`resultset`。

然后，我们将在一个`while`循环中调用`mysql_fetch_row`函数，每次从`resultset`中提取一行；也就是说，`resultset`中的第一行将被访问并分配给数组`row`。

回想一下，`users`表包含以下列：

+   `email_address varchar(30)`

+   `password varchar(30)`

+   `address_of_delivery text`

因此，数组`row`将包含访问行的完整信息，其中索引`row[0]`将包含`email_address`列的数据，`row[1]`将包含密码列的数据，`row[2]`将包含`address_of_delivery`列的数据。通过分别显示索引`row[0]`、`row[1]`和`row[2]`，将显示整行的信息。

最后，我们将调用`mysql_free_result`函数来释放分配给`resultset`的内存。然后，我们将调用`mysql_close`函数来关闭打开的连接处理器`conn`。

让我们打开 Cygwin 终端。我们需要两个终端窗口；在一个窗口中，我们将运行 SQL 命令，在另一个窗口中，我们将编译和运行 C 语言程序。通过按 *Alt+F2* 打开另一个终端窗口。在第一个终端窗口中，使用以下命令调用 MySQL 命令行：

```cpp
$ mysql -u root -p -h 127.0.0.1 
Enter password: 
Welcome to the MariaDB monitor.  Commands end with ; or \g. 
Your MySQL connection id is 27 
Server version: 5.7.14-log MySQL Community Server (GPL) 
Copyright (c) 2000, 2017, Oracle, MariaDB Corporation Ab and others. 
Type 'help;' or '\h' for help. Type '\c' to clear the current input statement. 
```

要与我们的 `ecommerce` 数据库一起工作，我们需要将其设置为当前数据库。因此，使用以下命令打开 `ecommerce` 数据库：

```cpp
MySQL [(none)]> use ecommerce; 
Reading table information for completion of table and column names 
You can turn off this feature to get a quicker startup with -A 
Database changed           
```

现在，`ecommerce` 是我们的当前数据库；也就是说，我们将执行的任何 SQL 命令都只应用于 `ecommerce` 数据库。让我们使用以下 SQL `SELECT` 命令来查看 `users` 数据库表中的现有行：

```cpp
MySQL [ecommerce]> select * from users; 
+---------------------+----------+------------------------------------+ 
| email_address       | password | address_of_delivery  |
+---------------------+----------+------------------------------------+
| bmharwani@yahoo.com | gold     | 11 Hill View Street, New York, USA

| harwanibm@gmail.com | diamond  | House No. xyz, Pqr Apartments, Uvw Lane, Mumbai, Maharashtra                        |                                                                                 | bintu@gmail.com     | platinum | abc Sea View, Ocean Lane, Opposite Mt. Everest, London, UKg 
+---------------------+----------+------------------------------------+
3 rows in set (0.00 sec)     
```

给定的输出显示 `users` 表中有三行。

要编译 C 程序，切换到第二个终端窗口。让我们使用 GCC 编译 `searchuser.c` 程序，如下所示：

```cpp
$ gcc searchuser.c -o searchuser -I/usr/local/include/mysql -L/usr/local/lib/mysql -lmysqlclient         
```

如果没有错误或警告，这意味着 `searchuser.c` 程序已编译成可执行文件，名为 `searchuser.exe`。让我们运行这个可执行文件：

```cpp
$ ./searchuser 
Enter email address to search: bmharwani@yahoo.com 
The details of the user with this email address are as follows: 
Email Address:bmharwani@yahoo.com
Password: gold 
Address of delivery: 11 Hill View Street, New York, USA 
```

哇！我们可以看到，带有电子邮件地址 bmharwani@yahoo.com 的用户完整信息显示在屏幕上。

现在，让我们继续下一个菜谱！

# 更新数据库中的信息

在这个菜谱中，我们将学习如何在数据库表中更新信息。我们假设已经存在一个 `users` 数据库表，包含三个列——`email_address`、`password` 和 `address_of_delivery`（请参阅本章开头，我们学习了如何创建数据库和其中的表）。输入电子邮件地址后，将显示用户的全部当前信息（即他们的密码和送货地址）。之后，用户将被提示输入新的密码和送货地址。这些新信息将更新到表中的当前信息。

# 如何做到这一点…

1.  初始化一个 `MYSQL` 对象：

```cpp
mysql_init(NULL);
```

1.  建立与指定主机上运行的 MySQL 服务器之间的连接。同时，生成一个连接处理器。如果建立与 MySQL 服务器引擎或 `ecommerce` 数据库的连接时发生错误，程序将终止：

```cpp
 if (!mysql_real_connect(conn, server, user, password, database, 0, NULL, 0)) 
 {
      fprintf(stderr, "%s\n", mysql_error(conn));
      exit(1);
 }
```

1.  输入需要更新信息的用户的电子邮件地址：

```cpp
printf("Enter email address of the user to update: ");
scanf("%s", emailaddress);
```

1.  创建一个 SQL `SELECT` 语句，该语句将搜索 `users` 表中与用户输入的电子邮件地址匹配的行：

```cpp
 strcpy(sqlquery,"SELECT * FROM users where email_address like \'");
 strcat(sqlquery,emailaddress);
 strcat(sqlquery,"\'");
```

1.  执行 SQL `SELECT` 语句。如果 SQL 查询未成功执行或发生其他错误，程序将终止：

```cpp
if (mysql_query(conn, sqlquery) != 0) 
{ 
     fprintf(stderr, "No row found in the users table with this          email address\n"); 
     exit(1); 
 }  
```

1.  如果 SQL 查询成功执行，则将匹配提供的电子邮件地址的行检索并分配给 `resultset`：

```cpp
 resultset = mysql_store_result(conn);
```

1.  检查 `resultset` 中是否至少有一行：

```cpp
if(mysql_num_rows(resultset) >0)
```

1.  如果 `resultset` 中没有行，则显示消息，指出在 `users` 表中没有找到指定电子邮件地址的行，并退出程序：

```cpp
printf("No user found with this email address\n");
```

1.  如果`resultset`中存在任何行，则访问它并将其分配给数组行：

```cpp
row = mysql_fetch_row(resultset)
```

1.  用户信息（即分配给子脚标`row[0]`、`row[1]`和`row[2]`的电子邮件地址、密码和送货地址）将在屏幕上显示：

```cpp
printf("Email Address: %s \n", row[0]);
printf("Password: %s \n", row[1]);
printf("Address of delivery: %s \n", row[2]);
```

1.  释放分配给`resultset`的内存：

```cpp
mysql_free_result(resultset);
```

1.  输入用户的新更新信息；即新的密码和新的送货地址：

```cpp
printf("Enter new password: ");
scanf("%s", upassword);
printf("Enter new address of delivery: ");
getchar();
gets(deliveryaddress);
```

1.  准备了一个包含新输入的密码和送货地址信息的 SQL `UPDATE`语句：

```cpp
strcpy(sqlquery,"UPDATE users set password=\'");
strcat(sqlquery,upassword);
strcat(sqlquery,"\', address_of_delivery=\'");
strcat(sqlquery,deliveryaddress);
strcat(sqlquery,"\' where email_address like \'");
strcat(sqlquery,emailaddress);
strcat(sqlquery,"\'");
```

1.  执行 SQL `UPDATE`语句。如果在执行 SQL `UPDATE`查询时发生任何错误，程序将终止：

```cpp
if (mysql_query(conn, sqlquery) != 0)                 
{                                                                                                                                                  fprintf(stderr, "The desired row in users table could not be 
    updated\n");  
    exit(1);
 }  
```

1.  如果 SQL `UPDATE`语句执行成功，将在屏幕上显示一条消息，告知用户信息已成功更新：

```cpp
printf("The information of user is updated successfully in users table\n");
```

1.  关闭打开的连接句柄：

```cpp
mysql_close(conn);
```

更新 MySQL 数据库表特定行的`updateuser.c`程序如下所示：

```cpp
#include <mysql/mysql.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void main() {
    MYSQL *conn;
    MYSQL_RES *resultset;
    MYSQL_ROW row;
    char *server = "127.0.0.1";
    char *user = "root";
    char *password = "Bintu2018$";
    char *database = "ecommerce";
    char emailaddress[30], sqlquery[255],             
    upassword[30],deliveryaddress[255];
    conn = mysql_init(NULL);
    if (!mysql_real_connect(conn, server, user, password, database, 0,     NULL, 0)) {
        fprintf(stderr, "%s\n", mysql_error(conn));
        exit(1);
    }
    printf("Enter email address of the user to update: ");
    scanf("%s", emailaddress);
    strcpy(sqlquery,"SELECT * FROM users where email_address like \'");
    strcat(sqlquery,emailaddress);
    strcat(sqlquery,"\'");
    if (mysql_query(conn, sqlquery) != 0)                 
    {                                                                                 
        fprintf(stderr, "No row found in the users table with this 
        email address\n");                                                                                                     
        exit(1);                                                                     
    }  
    resultset = mysql_store_result(conn);
    if(mysql_num_rows(resultset) >0)
    {
        printf("The details of the user with this email address are as 
        follows:\n");
        while ((row = mysql_fetch_row(resultset)) != NULL)
        {
            printf("Email Address: %s \n", row[0]);
            printf("Password: %s \n", row[1]);
            printf("Address of delivery: %s \n", row[2]);
        }
        mysql_free_result(resultset);
        printf("Enter new password: ");
        scanf("%s", upassword);
        printf("Enter new address of delivery: ");
        getchar();
        gets(deliveryaddress);
        strcpy(sqlquery,"UPDATE users set password=\'");
        strcat(sqlquery,upassword);
        strcat(sqlquery,"\', address_of_delivery=\'");
        strcat(sqlquery,deliveryaddress);
        strcat(sqlquery,"\' where email_address like \'");
        strcat(sqlquery,emailaddress);
        strcat(sqlquery,"\'");
        if (mysql_query(conn, sqlquery) != 0)                 
        {                                                                                                                                                         
            fprintf(stderr, "The desired row in users table could not 
            be updated\n");                                                             
            exit(1);                                                                     
        }  
        printf("The information of user is updated successfully in 
        users table\n");
    }
    else
        printf("No user found with this email address\n");
    mysql_close(conn);
}
```

现在，让我们深入了解代码背后的原理。

# 它是如何工作的...

在这个程序中，我们首先要求用户输入他们想要更新的电子邮件地址。然后，我们在`users`表中搜索是否存在具有匹配电子邮件地址的行。如果我们找到了它，我们显示用户的当前信息；即当前的电子邮件地址、密码和送货地址。之后，我们要求用户输入新的密码和新的送货地址。新的密码和送货地址将替换旧的密码和送货地址，从而更新`users`表。

我们将首先调用`mysql_init`函数，通过名称`conn`初始化一个`MYSQL`对象。然后，我们将`MYSQL`对象`conn`传递给`mysql_real_connect`函数，以建立与在指定主机上运行的 MySQL 服务器的连接。还将向`mysql_real_connect`函数传递其他几个参数，包括有效的用户 ID、密码、主机详情以及我们想要工作的数据库。`mysql_real_connect`函数将建立与在指定主机上运行的 MySQL 服务器的连接，并将`MYSQL`对象`conn`声明为连接句柄。这意味着`conn`可以在任何使用它的地方连接到`MySQL`服务器和`ecommerce`数据库。

如果在建立与 MySQL 服务器引擎或`ecommerce`数据库的连接时发生错误，程序将在显示错误消息后终止。如果成功建立与 MySQL 数据库引擎的连接，您将被提示输入您想要更新的用户记录的电子邮件地址。

正如我们之前提到的，我们首先将显示当前用户的信息。因此，我们将创建一个 SQL `SELECT`语句，并将在`users`表中搜索与用户输入的电子邮件地址匹配的行。然后，我们将调用`mysql_query`函数，并将创建的 SQL `SELECT`语句及其连接处理程序`conn`传递给它。

如果 SQL 查询没有成功执行或发生其他错误，程序将在显示错误消息后终止。如果查询成功执行，则通过调用`mysql_use_result`函数检索的结果行（即与提供的电子邮件地址匹配的行）将被分配给`resultset`。

然后，我们将调用`mysql_num_rows`函数以确保`resultset`中至少有一行。如果`resultset`中没有行，这意味着在`users`表中没有找到与给定电子邮件地址匹配的行。在这种情况下，程序将在通知在`users`表中没有找到给定电子邮件地址的行后终止。如果`resultset`中甚至有一行，我们将对`resultset`调用`mysql_fetch_row`函数，这将从一个`resultset`中提取一行并将其分配给数组行。

`users`表包含以下三个列：

+   `email_address varchar(30)`

+   `password varchar(30)`

+   `address_of_delivery text`

数组行将包含访问行的信息，其中子索引`row[0]`、`row[1]`和`row[2]`将分别包含`email_address`、`password`和`address_of_delivery`列的数据。通过显示分配给上述子索引的信息来显示当前用户的信息。然后，我们将调用`mysql_free_result`函数来释放分配给`resultset`的内存。

在此阶段，将要求用户输入新的密码和新的送货地址。我们将准备一个包含新输入的密码和送货地址信息的 SQL `UPDATE`语句。将调用`mysql_query`函数，并将 SQL `UPDATE`语句及其连接处理程序`conn`传递给它。

如果在执行 SQL `UPDATE`查询时发生任何错误，将再次显示错误消息，并终止程序。如果 SQL `UPDATE`语句成功执行，将显示一条消息，告知用户信息已成功更新。最后，我们将调用`mysql_close`函数来关闭打开的连接处理程序`conn`。

让我们打开 Cygwin 终端。我们需要两个终端窗口；在一个窗口中运行 SQL 命令，在另一个窗口中编译和运行 C。通过按*Alt+F2*打开另一个终端窗口。在第一个终端窗口中，使用以下命令调用 MySQL 命令行：

```cpp
$ mysql -u root -p -h 127.0.0.1 
Enter password: 
Welcome to the MariaDB monitor.  Commands end with ; or \g. 
Your MySQL connection id is 27 
Server version: 5.7.14-log MySQL Community Server (GPL) 
Copyright (c) 2000, 2017, Oracle, MariaDB Corporation Ab and others. 
Type 'help;' or '\h' for help. Type '\c' to clear the current input statement. 
```

要与我们的 `ecommerce` 数据库一起工作，我们需要将其设置为当前数据库。因此，使用以下命令打开 `ecommerce` 数据库：

```cpp
MySQL [(none)]> use ecommerce; 
Reading table information for completion of table and column names 
You can turn off this feature to get a quicker startup with -A 
Database changed            
```

现在，`ecommerce` 是我们的当前数据库；也就是说，我们将执行的任何 SQL 命令都只应用于 `ecommerce` 数据库。让我们使用以下 SQL `SELECT` 命令来查看 `users` 数据库表中的现有行：

```cpp
MySQL [ecommerce]> select * from users;
+---------------------+----------+------------------------------------+
| email_address       | password | address_of_delivery|
+---------------------+----------+------------------------------------+
| bmharwani@yahoo.com | gold     | 11 Hill View Street, New York, USA|
| harwanibm@gmail.com | diamond  | House No. xyz, Pqr Apartments, Uvw Lane, Mumbai, Maharashtra|
| bintu@gmail.com     | platinum | abc Sea View, Ocean Lane, Opposite Mt. Everest, London, UKg
+---------------------+----------+------------------------------------+
3 rows in set (0.00 sec)      
```

从前面的输出中我们可以看到，`users` 表中有三行。要编译 C 程序，切换到第二个终端窗口。让我们使用 GCC 编译 `updateuser.c` 程序，如下所示：

```cpp
$ gcc updateuser.c -o updateuser -I/usr/local/include/mysql -L/usr/local/lib/mysql -lmysqlclient           
```

如果没有错误或警告，这意味着 `updateuser.c` 程序已编译成可执行文件 `updateuser.exe`。让我们运行这个可执行文件：

```cpp
$ ./updateuser 
Enter email address of the user to update: harwanibintu@gmail.com 
No user found with this email address                     
```

让我们再次运行程序并输入一个已存在的电子邮件地址：

```cpp
$ ./updateuser 
Enter email address of the user to update: bmharwani@yahoo.com 
The details of the user with this email address are as follows: 
Email Address: bmharwani@yahoo.com 
Password: gold 
Address of delivery: 11 Hill View Street, New York, USA 
Enter new password: coffee 
Enter new address of delivery: 444, Sky Valley, Toronto, Canada 
The information of user is updated successfully in users table                 
```

因此，我们已经更新了电子邮件地址为 bmharwani@yahoo.com 的用户的行。为了确认该行已在 `users` 数据库表中更新，切换到运行 MySQL 命令行的终端窗口，并执行以下 SQL `SELECT` 命令：

```cpp
MySQL [ecommerce]> MySQL [ecommerce]> select * from users;
+---------------------+----------+------------------------------------+ 
| email_address       | password | address_of_delivery|
+---------------------+----------+------------------------------------+ 
| bmharwani@yahoo.com | coffee   | 444, Sky Valley, Toronto, Canada 
| 
| harwanibm@gmail.com | diamond  | House No. xyz, Pqr Apartments, Uvw Lane, Mumbai, Maharashtra 
|
| bintu@gmail.com     | platinum | abc Sea View, Ocean Lane, Opposite Mt. Everest, London, UKg
+---------------------+----------+------------------------------------+
```

*哇*！我们可以看到，电子邮件地址为 `bmharwani@yahoo.com` 的 `users` 表的行已被更新，并显示了新的信息。

现在，让我们继续下一个教程！

# 使用 C 从数据库中删除数据

在本教程中，我们将学习如何从数据库表中删除信息。我们假设已经存在一个包含三个列的 `users` 表，分别是 `email_address`、`password` 和 `address_of_delivery`（请参阅本章开头，我们在这里创建了一个 `ecommerce` 数据库和其中的 `users` 表）。您将被提示输入要删除行的用户的电子邮件地址。输入电子邮件地址后，将显示该用户的所有信息。之后，您将再次被要求确认是否要删除显示的行。确认后，该行将从表中永久删除。

# 如何操作...

1.  初始化一个 `MYSQL` 对象：

```cpp
mysql_init(NULL);
```

1.  建立与指定主机上运行的 MySQL 服务器连接。同时，生成一个连接处理程序。如果在建立与 MySQL 服务器引擎的连接过程中发生任何错误，程序将终止：

```cpp
  if (!mysql_real_connect(conn, server, user, password, database, 0, 
    NULL, 0)) {
      fprintf(stderr, "%s\n", mysql_error(conn));
      exit(1);
  }
```

1.  如果成功建立了与 MySQL 数据库引擎的连接，您将被提示输入要删除记录的用户的电子邮件地址：

```cpp
 printf("Enter email address of the user to delete: ");
 scanf("%s", emailaddress);
```

1.  创建一个 SQL `SELECT` 语句，该语句将搜索与用户输入的电子邮件地址匹配的 `users` 表中的行：

```cpp
 strcpy(sqlquery,"SELECT * FROM users where email_address like \'");
 strcat(sqlquery,emailaddress);
 strcat(sqlquery,"\'");
```

1.  执行 SQL `SELECT` 语句。如果 SQL 查询执行不成功，程序将在显示错误信息后终止：

```cpp
 if (mysql_query(conn, sqlquery) != 0)                 
 {                                                                                                                                   
    fprintf(stderr, "No row found in the users table with this email 
    address\n");                                                                                                     
    exit(1);                                                                     
 }  
```

1.  如果查询执行成功，则将检索与提供的电子邮件地址匹配的结果行（如果有的话），并将其分配给 `resultset`：

```cpp
resultset = mysql_store_result(conn);
```

1.  调用 `mysql_num_rows` 函数以确保 `resultset` 中至少有一行：

```cpp
if(mysql_num_rows(resultset) >0)
```

1.  如果 `resultset` 中没有行，这意味着在 `users` 表中没有找到与给定电子邮件地址匹配的行；因此，程序将终止：

```cpp
printf("No user found with this email address\n");
```

1.  如果结果集中有任何行，则该行将从 `resultset` 中提取出来，并将分配给数组行：

```cpp
row = mysql_fetch_row(resultset)
```

1.  通过显示数组行中的相应子脚本来显示用户信息：

```cpp
printf("Email Address: %s \n", row[0]);
printf("Password: %s \n", row[1]);
printf("Address of delivery: %s \n", row[2]);
```

1.  分配给 `resultset` 的内存被释放：

```cpp
mysql_free_result(resultset);The user is asked whether he/she really want to delete the shown record.
printf("Are you sure you want to delete this record yes/no: ");
scanf("%s", k);
```

1.  如果用户输入 `yes`，将创建一个 SQL `DELETE` 语句，该语句将从 `users` 表中删除与指定电子邮件地址匹配的行：

```cpp
if(strcmp(k,"yes")==0)
{
    strcpy(sqlquery, "Delete from users where email_address like 
    \'");
    strcat(sqlquery,emailaddress);
    strcat(sqlquery,"\'");
```

1.  执行 SQL `DELETE` 语句。如果在执行 SQL `DELETE` 查询时发生任何错误，程序将终止：

```cpp
if (mysql_query(conn, sqlquery) != 0)                 
{                                                                                   
    fprintf(stderr, "The user account could not be deleted\n");                                                             
    exit(1);                                                                     
}
```

1.  如果 SQL `DELETE` 语句执行成功，将显示一条消息，告知指定电子邮件地址的用户账户已成功删除：

```cpp
printf("The user with the given email address is successfully deleted from the users table\n");
```

1.  打开的连接处理程序被关闭：

```cpp
mysql_close(conn);
```

用于从 MySQL 数据库表中删除特定行的 `deleteuser.c` 程序如下所示：

```cpp
#include <mysql/mysql.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void main() {
MYSQL *conn;
MYSQL_RES *resultset;
MYSQL_ROW row;
char *server = "127.0.0.1";
char *user = "root";
char *password = "Bintu2018$";
char *database = "ecommerce";
char emailaddress[30], sqlquery[255],k[10];
conn = mysql_init(NULL);
if (!mysql_real_connect(conn, server, user, password, database, 0, NULL, 0)) {
    fprintf(stderr, "%s\n", mysql_error(conn));
    exit(1);
}
printf("Enter email address of the user to delete: ");
scanf("%s", emailaddress);
strcpy(sqlquery,"SELECT * FROM users where email_address like \'");
strcat(sqlquery,emailaddress);
strcat(sqlquery,"\'");
if (mysql_query(conn, sqlquery) != 0)                 
{                                                                          
    fprintf(stderr, "No row found in the users table with this email 
    address\n");                                                             
    exit(1);                                                                      
}  
resultset = mysql_store_result(conn);
if(mysql_num_rows(resultset) >0)
{
    printf("The details of the user with this email address are as 
    follows:\n");
    while ((row = mysql_fetch_row(resultset)) != NULL)
    {
        printf("Email Address: %s \n", row[0]);
        printf("Password: %s \n", row[1]);
        printf("Address of delivery: %s \n", row[2]);
    }
    mysql_free_result(resultset);
    printf("Are you sure you want to delete this record yes/no: ");
    scanf("%s", k);
    if(strcmp(k,"yes")==0)
    {
        strcpy(sqlquery, "Delete from users where email_address like 
        \'");
        strcat(sqlquery,emailaddress);
        strcat(sqlquery,"\'");
        if (mysql_query(conn, sqlquery) != 0)                 
        {                                                                                 
            fprintf(stderr, "The user account could not be deleted\n");                                                             
            exit(1);                                                                      
        }  
        printf("The user with the given email address is successfully 
        deleted from the users table\n");
    }
}
else
    printf("No user found with this email address\n");
    mysql_close(conn);
}
```

现在，让我们深入了解代码，以更好地理解它。

# 它是如何工作的...

我们将首先调用 `mysql_init` 函数，通过名称 `conn` 初始化一个 `MYSQL` 对象。然后，我们将 `MYSQL` 对象 `conn` 传递给 `mysql_real_connect` 函数，该函数用于建立与在指定主机上运行的 MySQL 服务器的连接。还将向 `mysql_real_connect` 函数传递其他几个参数，包括有效的用户 ID、密码、主机详细信息以及我们想要工作的数据库。`mysql_real_connect` 函数将建立与在指定主机上运行的 MySQL 服务器的连接，并将一个 `MYSQL` 对象 `conn` 声明为连接处理程序。这意味着 `conn` 可以在任何使用它的地方连接到 MySQL 服务器和 commerce 数据库。

如果在连接到 MySQL 服务器引擎或 `ecommerce` 数据库时发生错误，程序将在显示错误消息后终止。如果成功连接到 MySQL 数据库引擎，系统将提示您输入要删除记录的用户电子邮件地址。

我们首先将显示用户的信息，然后将从用户那里获取是否真的想要删除该行的许可。因此，我们将创建一个 SQL `SELECT` 语句，该语句将搜索与用户输入的电子邮件地址匹配的 `users` 表中的行。然后，我们将调用 `mysql_query` 函数，并将创建的 SQL `SELECT` 语句及其连接处理程序 `conn` 传递给它。

如果 SQL 查询没有成功执行或发生其他错误，程序将在显示错误消息后终止。如果查询执行成功，则通过调用 `mysql_use_result` 函数检索到的结果行（即与提供的电子邮件地址匹配的行）将被分配给 `resultset`。

我们将调用 `mysql_num_rows` 函数以确保 `resultset` 中至少有一行。如果没有行在 `resultset` 中，这意味着在 `users` 表中没有找到与给定电子邮件地址匹配的行。在这种情况下，程序将在告知在 `users` 表中没有找到给定电子邮件地址的行后终止。如果 `resultset` 中甚至有一行，我们将对 `resultset` 调用 `mysql_fetch_row` 函数，这将从一个 `resultset` 中提取一行并将其分配给数组行。

`users` 表包含以下三个列：

+   `email_address varchar(30)`

+   `password varchar(30)`

+   `address_of_delivery text`

数组行将包含访问行的信息，其中子索引 `row[0]`、`row[1]` 和 `row[2]` 分别包含 `email_address`、`password` 和 `address_of_delivery` 列的数据。当前用户信息将通过显示分配给子索引 `row[0]`、`row[1]` 和 `row[2]` 的当前电子邮件地址、密码和送货地址来显示。然后，我们将调用 `mysql_free_result` 函数来释放分配给 `resultset` 的内存。

在此阶段，将要求用户确认他们是否真的想要删除显示的记录。用户应输入全部小写的 `yes` 来删除记录。如果用户输入 `yes`，将创建一个 SQL `DELETE` 语句，该语句将删除与指定电子邮件地址匹配的 `users` 表中的行。将调用 `mysql_query` 函数，并将 SQL `DELETE` 语句及其连接处理器 `conn` 传递给它。

如果在执行 SQL `DELETE` 查询时发生任何错误，将再次显示错误消息，并且程序将终止。如果 SQL `DELETE` 语句执行成功，将显示一条消息，告知指定邮件地址的用户账户已成功删除。最后，我们将调用 `mysql_close` 函数来关闭已打开的连接处理器 `conn`。

让我们打开 Cygwin 终端。我们需要两个终端窗口；在一个窗口中，我们将运行 MySQL 命令，在另一个窗口中，我们将编译和运行 C 语言。通过按 *Alt+F2* 打开另一个终端窗口。在第一个终端窗口中，通过以下命令调用 MySQL 命令行：

```cpp
$ mysql -u root -p -h 127.0.0.1 
Enter password: 
Welcome to the MariaDB monitor.  Commands end with ; or \g. 
Your MySQL connection id is 27 
Server version: 5.7.14-log MySQL Community Server (GPL) 
Copyright (c) 2000, 2017, Oracle, MariaDB Corporation Ab and others. 
Type 'help;' or '\h' for help. Type '\c' to clear the current input statement. 
```

要与我们的 `ecommerce` 数据库一起工作，我们需要将其设置为当前数据库。因此，使用以下命令打开 `ecommerce` 数据库：

```cpp
MySQL [(none)]> use ecommerce; 
Reading table information for completion of table and column names 
You can turn off this feature to get a quicker startup with -A 
Database changed            
```

现在，`ecommerce` 是我们的当前数据库；也就是说，我们将执行的任何 SQL 命令都只应用于 `ecommerce` 数据库。让我们使用以下 SQL `SELECT` 命令来查看 `users` 数据库表中的现有行：

```cpp
MySQL [ecommerce]> select * from users;
+---------------------+----------+------------------------------------+
| email_address | password | address_of_delivery | 
+---------------------+----------+------------------------------------+
| bmharwani@yahoo.com | coffee | 444, Sky Valley, Toronto, Canada 
|
| harwanibm@gmail.com | diamond | House No. xyz, Pqr Apartments, Uvw Lane, Mumbai, Maharashtra | 
| bintu@gmail.com | platinum | abc Sea View, Ocean Lane, Opposite Mt. Everest, London, UKg
+---------------------+----------+------------------------------------+
3 rows in set (0.00 sec)
```

从前面的输出中，我们可以看到 `users` 表中有三行。要编译 C 程序，切换到第二个终端窗口。让我们使用 GCC 编译 `deleteuser.c` 程序，如下所示：

```cpp
$ gcc deleteuser.c -o deleteuser -I/usr/local/include/mysql -L/usr/local/lib/mysql -lmysqlclient
```

如果你没有收到任何错误或警告，这意味着 `deleteuser.c` 程序已编译成可执行文件，名为 `deleteuser.exe`。让我们运行这个可执行文件：

```cpp
$ ./deleteuser
Enter email address of the user to delete: harwanibintu@gmail.com 
No user found with this email address                
```

现在，让我们使用有效的电子邮件地址再次运行程序：

```cpp
$ ./deleteuser 
Enter email address of the user to delete: bmharwani@yahoo.com 
The details of the user with this email address are as follows:
Email Address: bmharwani@yahoo.com
Password: coffee
Address of delivery: 444, Sky Valley, Toronto, Canada
Are you sure you want to delete this record yes/no: yes 
The user with the given email address is successfully deleted from the users table
```

因此，具有电子邮件地址 `bmharwani@yahoo.com` 的用户行将从 `users` 表中删除。为了确认该行已从 `users` 数据库表中删除，切换到运行 MySQL 命令行的终端窗口，并执行以下 SQL `SELECT` 命令：

```cpp
 MySQL [ecommerce]> select * from users;
+---------------------+----------+------------------------------------+
| email_address       | password | address_of_delivery 
| 
+---------------------+----------+------------------------------------+
| harwanibm@gmail.com | diamond  | House No. xyz, Pqr Apartments, Uvw Lane, Mumbai, Maharashtra 
| 
| bintu@gmail.com     | platinum | abc Sea View, Ocean Lane, Opposite Mt. Everest, London, UKg 
+---------------------+----------+------------------------------------+
```

*Voila*！我们可以看到现在 `users` 表中只剩下两行，这证实了一行已从 `users` 表中删除。
