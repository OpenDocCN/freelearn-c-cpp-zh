# Remote Databases with Qt SQL

Qt SQL is not dependent on any particular database driver. The same API can be used with various popular database backends. Databases can get huge storage, whereas mobile and embedded devices have limited amounts of storage, more so with embedded devices than mobile phones. You will learn about using Qt to access databases remotely over the network.

We will cover the following topics in this chapter:

*   Drivers
*   Connecting to database
*   Creating a database
*   Adding to a database

# Technical requirements

You can grab this chapter's source code in the `cp10` branch at `git clone -b cp10 https://github.com/PacktPublishing/Hands-On-Mobile-and-Embedded-Development-with-Qt-5`.

You should also have installed the `sqlite` or `mysql` package for your system.

# Drivers are database backends

Qt supports a variety of database drivers or backends to the databases. The backends wrap the various system databases and allow Qt to have a unified API frontend. Qt supports the following database types:

| Database types | Software |
| QDB2 | IBM Db2 |
| QIBASE | Borland InterBase |
| QMYSQL | MySQL |
| QOCI | Oracle Call Interface |
| QODBC | ODBC |
| QPSQL | PostgreSQL |
| QSQLITE | SQLite version 3 or above |
| QSQLITE2 | SQLite version 2 |
| QTDS | Sybase Adaptive Server |

We will be looking into QMYSQL type, since it supports remote access. MySQL can be installed on Raspberry Pi. QSQLITE3 can be shared on a network resource and made to support remote access, and iOS and Android have support for SQLite.

# Setup

The MySQL database will need to be configured to let you have remote access to it. Let's look at how we can do this:

1.  You will need to have the server and/or client installed.
2.  Then, we'll create the database and make it accessible from the network, if needed. This will be done using the command line and a Terminal application.

# The MySQL server

I am using Ubuntu, so these commands will be mostly specific to a Debian-based Linux. If you are using a different Linux distribution, only the installation command would be different. You should install the MySQL server and client according to your distribution. The commands to create the database would be the same.

Here's how we will set up the server:

1.  You will need the MySQL server and client installed:

`sudo apt-get install mysql-server mysql-client`

2.  Run `sudo mysql_secure_installation`, which will allow you to set up the root account. Then, log in to the `mysql` root account:

`sudo mysql -u root -p`

3.  Create a new database `username` and `password`: `GRANT ALL PRIVILEGES ON *.* TO 'username'@'localhost' IDENTIFIED BY 'password';`. Change `username` to your database user, and `password` to a password you want to use to access this database.

4.  To make the server accessible from a host other than localhost, edit `/etc/mysql/mysql.conf.d/mysqld.cnf`.
5.  Change the `bind-address = localhost to bind-address = <your ip>` line, `<your ip>` being the IP address of the machine that the database is on. Then, restart the `mysql` server:

`sudo  service mysql restart`

Back in your MySQL console, let a remote user access the database:

`GRANT ALL ON *.* TO 'username'@'<your ip>' IDENTIFIED BY 'password';`

Change `<your ip>` to the IP address or hostname of the client device, `username` to the username you used on the MySQL server, and `password` to the password they will use.

# SQLite

SQLite is a file-based database and, as such, there is no such thing as a server. We can still connect to it remotely via a network filesystem, such as Windows file sharing/Samba, **Network File System** (NFS), or the **Secure Shell File System** (**SSHFS**) on Linux. SSHFS allows you to mount and access a remote filesystem like it is a local filesystem. 

There's no need to create a database manually using arcane commands unless you need to, as we will create it using Qt!

On Android, there are Samba clients, which will mount a Windows network share so we can use that. If you are using a Raspberry Pi or some other development board, you might be able to use SSHFS to mount a remote directory over SSH.

# Connecting to a local or remote database

Once we have the database configured and running, we can now connect to it using the same functions regardless of whether it is local or a remote database. Now, let's take a look at writing code to connect to a database, whether local or remote.

Databases are either locally available, which usually means on the same machine, or accessed remotely over a network. Connecting to these different databases using Qt is essentially the same. Not all databases support remote access. 

Let's begin by using a local database.

To use the `sql` module, we need to add `sql` to the profile:

`QT += sql`

To connect to a database in Qt, we need to use the `QSqlDatabase` class.

# QSqlDatabase

Despite the name, `QSqlDatabase` represents a connection to a database, not the database itself.

To create a connection to a database, you first need to specify which database type you are using. It is referenced as a string representation of the supported database. Let's first choose the MySQL `QMYSQL` database.

The source code can be found on the Git repository under the `Chapter10-1` directory, in the `cp10` branch.

To use `QSqlDatabase`, we first need to add the database to create its instance.

The static `QSqlDatbase::addDatabase` function takes one parameter, that of a database type, and adds the instance of the database to the list of database connections.

Here, we add a MySQL database, so use the `QMYSQL` type:

```cpp
 QSqlDatabase db = QSqlDatabase::addDatabase("QMYSQL");
```

If you are connecting to a SQLite database, use the `MSQLITE` database type:

```cpp
QSqlDatabase db = QSqlDatabase::addDatabase("MSQLITE");
```

Most databases require a username and password. To set the `username` and `password`, use the following:

```cpp
db.setUserName("username");
db.setPassword("password");
```

Since we are connecting to a remote MySQL database, we need to specify the hostname as well. It can also be an IP address:

```cpp
db.setHostName("10.0.0.243");
```

To start the connection, call the `open()` function:

```cpp
bool ok = db.open()
```

`open` returns a `bool`, which is `true` if it was successful, or `false` if it failed, in which case we can check the error:

```cpp
if (!db.open()) {
    qWarning() << dq.lastError.text();
}
```

If this opens successfully, we are connected to the database.

Let's actually create the remote database, since we have the needed permissions.

# Creating and opening a database

For the SQLite database, once we open it, it creates the database on the filesystem. For MySQL, we have to send MySQL commands to create the database. We construct the SQL query using `QSqlQuery` do this in MySQL. `QSqlQuery` takes the database object as an argument:

```cpp
QSqlQuery query(db);
```

To send a query, we call the `exec()` function on the `QSqlQuery` object. It takes a `String` as a typical `query` syntax:

```cpp
QString dbname = "MAEPQTdb";
if (!query.exec("CREATE DATABASE IF NOT EXISTS " + dbname)) {
    qWarning() << "Database query error" << query.lastError().text();
}
```

`dbname` here is any `String` we want the database name to be; I am using `MAEPQT db`.

If this command fails, we issue a warning message. If it succeeds, then we go on and issue the command to `USE` it, so we call another `query` command:

```cpp
query.exec("USE " + dbname);
```

From here, we need to create some tables. I will keep it simple and fill it with some data.

We start another query, but with an empty command, and the `db` object as the second argument, which will create the `QSqlQuery` object on the specified database, but does not execute any commands until we are ready:

```cpp
QSqlQuery q("", db);
q.exec("drop table Mobile");
q.exec("drop table Embedded");
q.exec("create table Mobile (id integer primary key, Device varchar,
Model varchar, Version number)");

q.exec("create table Embedded (id integer primary key, Device varchar,Model varchar, Version number)");

```

The database is prepared, so now we can add some data.

# Adding data to the database

The Qt documents state that it is not a best practice to keep the `QSqlDatabase` object around.

Here are a few different ways we could go about this:

1.  We could use `QSqlDatabase::database` to grab an instance of the opened database:

```cpp
QSqlDatabase db = QSqlDatabase::database("MAEPQTdb");
QSqlQuery q("", db);
q.exec("insert into Mobile values (0, 'iPhone', '6SE', '12.1.2')");
q.exec("insert into Mobile values (1, 'Moto', 'XT1710-09', '2')");
q.exec("insert into Mobile values (1, 'rpi', '1', '1')");
q.exec("insert into Mobile values (1, 'rpi', '2', '2')");
q.exec("insert into Mobile values (1, 'rpi', '3', '3')");
```

2.  We can also use another function of `QSqlQuery` , named `prepare()`, which prepares the query string for execution using a proxy variable.

Then, we can use `bindValue` to bind the value to its identifier:

```cpp
q.prepare("insert into Mobile values (id,  device, model, version)"
          "values ( :id, :device, :model, :version)");

q.bindValue(":id", 0);
q.bindValue(":device", "iPhone");
q.bindValue(":model", "6SE");
q.bindValue(":version", "12.1.2");
q.exec();
```

3.  As an alternative, you can call `bindValue` with the first argument being the index of the position of the identifier, starting at the number 0 and working upward through the values:

```cpp
q.bindValue(1, "iPhone");
q.bindValue(3, "12.1.2");
q.bindValue(2, "6SE");
```

4.  You can also use `bindValue` in the order of values:

```cpp
q.bindValue(0);
q.bindValue("iPhone");
q.bindValue("6SE");
q.bindValue("12.1.2");
```

Next, let's look at retrieving data from the database.

# Executing queries

So far, we have been running queries, but not getting any data in return. One of the points of a database is to query for data, not just to enter it. What fun would it be if we could only input data? The Qt API has a way to accommodate the different syntax and millions of ways a query can be made. Most of the time, it is specific to the type of data you need to be returned, but also specific to the database data itself. Luckily, `QSqlQuery` is general enough that the query parameter is a string.

# QSqlQuery

To retrieve data, execute a query using `QsqlQuery` and then operate on the records using the following functions:

*   `first()`
*   `last()`
*   `next()`
*   `previous()`
*   `seek(int)`

The `first()` and `last()` functions will retrieve the first and last records respectively. To iterate backward through the records, use `previous()`. The `seek (int)` function takes on integer as an argument to determine which record to retrieve.

We will use `next()`, which will iterate forward through the records found in the query:

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

We also use `value` to retrieve each field's data, which takes an `int` that indicates the position of the record starting at 0.

You can also use `QSqlRecord` and `QSqlField` to do the same thing, but with more clarity as to what is actually going on:

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

To get the record data, use `value()`, which will return a `QVariant` that represents the data for that record field.

We could have used a model-based widget and then used `QsqlQueryModel` to execute the query.

# QSqlQueryModel

 `QSqlQueryModel` inherits from `QSqlQuery`, and returns a model object that can be used with model-based widgets and other classes. If we change our `QTableWidget` to `QTableView`, we can use `QSqlQueryModel` to be its data model:

```cpp
QSqlQueryModel *model = new QSqlQueryModel;
model->setQuery("SELECT * FROM Mobile", db);
tableView->setModel(model);
```

Here is my Raspberry Pi (with a Lego stand!) running the database example using the MySQL plugin remotely:

![](img/0e6010a7-476f-4740-bd4c-a405253bbfda.jpg)

# Summary

In this chapter, we learned that `QSqlDatabase` represents a connection to any supported database. You can use this to log in to remote MySQL databases or a SQLite database on a network share. To perform data retrieval, use `QSqlQuery`. You use the same class to set data and tables and other database commands. You can use `QSqlDatabaseModel` if you are using a model-view application.

In the next chapter, we will explore in-app purchasing using the Qt Purchasing module.