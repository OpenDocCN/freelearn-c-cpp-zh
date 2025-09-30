# Using MySQL Database

MySQL is one of the most popular database management systems in recent times. Databases, as we all know, are used for storing data that's going to be used in the future when required. The data in a database can be secured through encryption and can be indexed for faster access. Where the volume of data is too high, a database management system is preferred over a traditional sequential and random file handling system. Storing data in a database is a very important task in any application.

This chapter is focused on understanding how table rows are managed in the database tables. In this chapter, you will learn about the following recipes:

*   Displaying all the built-in tables in a default MySQL database
*   Storing information into MySQL database
*   Searching desired information in the database
*   Updating information in the database
*   Deleting data from the database using C

We will review the most commonly used functions in MySQL before we move on to the recipes. Also, ensure that you read *Appendix B* and *Appendix C* to install Cygwin and MySQL Server before implementing the recipes in this chapter.

# Functions in MySQL

While accessing and working with MySQL database in C programming, we will have to use several functions. Let's go through them.

# mysql_init()

This initializes a `MYSQL` object that can be used in the `mysql_real_connect()` method. Here is its syntax:

```cpp
MYSQL *mysql_init(MYSQL *object)
```

If the object parameter that's passed is `NULL`, then the function initializes and returns a new object; otherwise, the supplied object is initialized and the address of the object is returned.

# mysql_real_connect()

This establishes a connection to a MySQL database engine running on the specified host. Here is its syntax:

```cpp
MYSQL *mysql_real_connect(MYSQL *mysqlObject, const char *hostName, const char *userid, const char *password, const char *dbase, unsigned int port, const char *socket, unsigned long flag)
```

Here:

*   `mysqlObject` represents the address of an existing `MYSQL` object.
*   `hostName` is where the hostname or IP address of the host is provided. To connect to a local host, either `NULL` or the string *localhost* is provided.
*   `userid` represents a valid MySQL login ID.
*   `password` represents the password of the user.
*   `dbase` represents the database name to which the connection has to be established.
*   `port` is where either value `0`  is specified or the port number for the TCP/IP connection is supplied.
*   `socket`is where either `NULL` is specified or the socket or named pipe is supplied.
*   `flag` can be used to enable certain features, such as handling expired passwords and applying compression in the client/server protocol, but its value is usually kept at `0`.

The function returns a `MYSQL` connection handler if the connection is established; otherwise, it returns `NULL`.

# mysql_query()

This function executes the supplied SQL query. Here is its syntax:

```cpp
int mysql_query(MYSQL *mysqlObject, const char *sqlstmt)
```

Here:

*   `mysqlObject` represents the `MYSQL` object
*   `sqlstmt` represents the null-terminated string that contains the SQL statement to be executed

The function returns `0` if the SQL statement executes successfully; otherwise, it returns a non-zero value.

# mysql_use_result()

After successful execution of an SQL statement, this method is used to save the result set. This means that the result set is retrieved and returned. Here is its syntax:

```cpp
MYSQL_RES *mysql_use_result(MYSQL *mysqlObject)
```

Here,  `mysqlObject` represents the connection handler.

If no error occurs, the function returns a `MYSQL_RES` result structure. In case of any error, the function returns `NULL`.

# mysql_fetch_row()

This function fetches the next row from a result set. The function returns `NULL` if there are no more rows in the result set to retrieve or if an error occurs. Here is its syntax:

```cpp
MYSQL_ROW mysql_fetch_row(MYSQL_RES *resultset)
```

Here, the `resultset` parameter is the set from which the next row has to be fetched. You can access values in the column of the row by using the subscript `row[0]`, `row[1]`, and so on, where `row[0]` represents the data in the first column, `row[1]` represents the data in the second column, and so on.

# mysql_num_fields()

This returns the number of values; that is, columns in the supplied row. Here is its syntax:

```cpp
unsigned int mysql_num_fields(MYSQL_ROW row)
```

Here, the parameter row represents the individual row that is accessed from the `resultset`.

# mysql_free_result()

This frees the memory allocated to a result set. Here is its syntax:

```cpp
void mysql_free_result(MYSQL_RES *resultset)
```

Here, `resultset` represents the set whose memory we want to free up.

# mysql_close()

This function closes the previously opened MySQL connection. Here is its syntax:

```cpp
void mysql_close(MYSQL *mysqlObject)
```

It de-allocates the connection handler that's represented by the `mysqlObject` parameter. The function returns no value.

This covers the functions that we need to know for using the MySQL database for our recipes. From the second recipe onward, we will be working on a database table. So, let's get started and create a database and a table inside it.

# Creating a MySQL database and tables

Open the Cygwin Terminal and open the MySQL command line by giving the following command. Through this command, we want to open MySQL through the user ID root and try to connect with the MySQL server running at the localhost (`127.0.0.1`):

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

The preceding MySQL prompt that appears confirms that the `userid` and `password` have been entered correctly and that you are successfully connected to a running MySQL server. Now, we can go ahead and run SQL commands.

# Create database

The `create database` statement creates the database with the specified name. Here is the syntax:

```cpp
Create database database_name;
```

Here, `database_name` is the name of the new database to be created.

Let's create a database by the name `ecommerce` for our recipes:

```cpp
MySQL [(none)]> create database ecommerce; 
Query OK, 1 row affected (0.01 sec)                                            
```

To confirm that our `ecommerce` database has been successfully created, we will use the `show databases` statement to see the list of existing databases on the MySQL server:

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

In the preceding database listing, we can see the name `ecommerce`, which confirms that our database has been successfully created. Now, we will apply the `use` statement to access the `ecommerce` database, as shown here:                          

```cpp
MySQL [(none)]> use ecommerce;
Database changed        
```

Now, the `ecommerce` database is in use, so whatever SQL commands we will give will be applied to the `ecommerce` database only. Next, we need to create a table in our `ecommerce` database. For creating a database table, the `Create table` command is used. Let's discuss it next.

# Create table

This creates a database table with the specified name. Here is its syntax:

```cpp
CREATE TABLE table_name (column_name column_type,column_name column_type,.....);
```

Here:

*   `table_name` represents the name of the table that we want to create.
*   `column_ name` represents the column names that we want in the table.
*   `column_type` represents the data type of the column. Depending on the type of data we want to store in the column, the `column_type` can be `int`, `varchar`, `date`, `text`, and so on.

The `create table` statement creates a `users` table with three columns: `email_address`, `password`, and `address_of_delivery`. Assuming that this table will contain information of the users who have placed orders online, we will be storing their email address, password, and the location where the order has to be delivered:

```cpp
MySQL [ecommerce]> create table users(email_address varchar(30), password varchar(30), address_of_delivery text);
Query OK, 0 rows affected (0.38 sec)                                           
```

To confirm that the table has been successfully created, we will use the `show tables` command to display the list of existing tables in the currently opened database, as shown here:

```cpp
MySQL [ecommerce]> show tables;
+---------------------+ 
| Tables_in_ecommerce | 
+---------------------+ 
| users               | 
+---------------------+ 
1 row in set (0.00 sec)         
```

The output of the `show tables` command displays the `users` table, thus confirming that the table has indeed been created successfully. To see the table structure (that is, its column names, column types, and column width), we will use the `describe` statement. The following statement displays the structure of the `users` table:

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

So, now that we have learned about some basic commands to work with our database, we can begin with the first recipe of this chapter.

# Displaying all the built-in tables in a default mysql database

The MySQL server, when installed, comes with certain default databases. One of those databases is `mysql`. In this recipe, we will learn to display all the table names that are available in the `mysql` database.

# How to do it...

1.  Create a MySQL object:

```cpp
mysql_init(NULL);
```

2.  Establish a connection to the MySQL server running at the specified host. Also, connect to the desired database:

```cpp
mysql_real_connect(conn, server, user, password, database, 0, NULL, 0)
```

3.  Create an execute SQL statement, comprised of `show tables`:

```cpp
mysql_query(conn, "show tables")
```

4.  Save the result of the executing SQL query (that is, the table information of the `mysql` database) into a `resultset`:

```cpp
res = mysql_use_result(conn);
```

5.  Fetch one row at a time from the `resultset` in a `while` loop and display only the table name from that row:

```cpp
while ((row = mysql_fetch_row(res)) != NULL)
     printf("%s \n", row[0]);
```

6.  Free up the memory that is allocated to the `resultset`:

```cpp
mysql_free_result(res);
```

7.  Close the opened connection handler:

```cpp
mysql_close(conn);
```

The `mysql1.c` program for displaying all the tables in the built-in `mysql` database is as follows:

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

Now, let's go behind the scenes to understand the code better.

# How it works...

We will start by establishing a connection with the MySQL server and for that, we need to invoke the `mysql_real_connect` function. But we have to pass a `MYSQL` object to the `mysql_real_connect` function and we have to invoke the `mysql_init` function to create the `MYSQL` object. Hence, the `mysql_init` function is first invoked to initialize a `MYSQL` object by the name `conn`.

We will then supply the `MYSQL` object `conn` to the `mysql_real_connect` function, along with the valid user ID, password, and the host details. The `mysql_real_connect` function will establish a connection to the MySQL server running at the specified host. Besides this, the function will link to the supplied `mysql` database and will declare `conn` as the connection handler. This means that `conn` will be used in the rest of the program whenever we want to perform some action to the specified MySQL server and the `mysql` database.

If any error occurs in establishing the connection to the MySQL database engine, the program will terminate after displaying an error message. If the connection to the MySQL database engine is established successfully, the `mysql_query` function is invoked and the SQL statement `show tables` and the connection handler `conn` are supplied to it. The `mysql_query` function will execute the supplied SQL statement. To save the resulting table information of the `mysql` database, the `mysql_use_result` function is invoked. The table information that's received from the `mysql_use_result` function will be assigned to `resultset` `res`.

Next, we will invoke the `mysql _fetch_row` function in a `while` loop that will extract one row at a time from the `resultset` `res`; that is, one table detail will be fetched at a time from the `resultset` and assigned to the array row. The array row will contain complete information of one table at a time. The table name stored in the `row[0]` subscript is displayed on the screen. With every iteration of the `while` loop, the next piece of table information is extracted from `resultset` `res` and assigned to the array row. Consequently, all the table names in the `mysql` database will be displayed on the screen.

Then, we will invoke the `mysql_free_result` function to free up the memory that is allocated to `resultset` `res` and, finally, we will invoke the `mysql_close` function to close the opened connection handler `conn`.

Let's use GCC to compile the `mysql1.c` program, as shown here:

```cpp
$ gcc mysql1.c -o mysql1 -I/usr/local/include/mysql -L/usr/local/lib/mysql -lmysqlclient          
```

If you get no errors or warnings, that means the `mysql1.c` program has compiled into an executable file, `mysql1.exe`. Let's run this executable file:

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

*Voila*! As you can see, the output shows the list of built-in tables in the `mysql` database. Now, let's move on to the next recipe!

# Storing information in MySQL database

In this recipe, we will learn how to insert a new row into the `users` table. Recall that at the beginning of this chapter, we created a database called `ecommerce`, and in that database, we created a table called `users` with the following columns:

```cpp
email_address varchar(30)
password varchar(30) 
address_of_delivery text  
```

We will be inserting rows into this `users` table now.

# How to do it…

1.  Initialize a MYSQL object:

```cpp
conn = mysql_init(NULL);
```

2.  Establish a connection to the MySQL server running at the localhost. Also, connect to the database that you want to work on:

```cpp
mysql_real_connect(conn, server, user, password, database, 0, NULL, 0)
```

3.  Enter the information of the new row that you want to insert into the `users` table in the `ecommerce` database, which will be for the new user's email address, password, and address of delivery:

```cpp
printf("Enter email address: ");
scanf("%s", emailaddress);
printf("Enter password: ");
scanf("%s", upassword);
printf("Enter address of delivery: ");
getchar();
gets(deliveryaddress);
```

4.  Prepare an SQL `INSERT` statement comprising this information; that is, the email address, password, and address of delivery of the new user:

```cpp
strcpy(sqlquery,"INSERT INTO users(email_address, password, address_of_delivery)VALUES (\'");
strcat(sqlquery,emailaddress);
strcat(sqlquery,"\', \'");
strcat(sqlquery,upassword);
strcat(sqlquery,"\', \'");
strcat(sqlquery,deliveryaddress);
strcat(sqlquery,"\')");
```

5.  Execute the SQL `INSERT` statement to insert a new row into the `users` table in the `ecommerce` database:

```cpp
mysql_query(conn, sqlquery)
```

6.  Close the connection handler:

```cpp
mysql_close(conn);
```

The `adduser.c` program for inserting a row into a MySQL database table is shown in the following code:

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

Now, let's go behind the scenes to understand the code better.

# How it works...

We start by invoking the `mysql_init` function to initialize a `MYSQL` object by the name `conn`. The initialized `MYSQL` object `conn` is then supplied for invoking the `mysql_real_connect` function, along with the valid user ID and password, which in turn will establish a connection to the MySQL server running on the localhost. In addition, the function will link to our `ecommerce` database.

If any error occurs in establishing the connection to the MySQL database engine, an error message will be displayed and the program will terminate. If the connection to the MySQL database engine is established successfully, then `conn` will act as a connection handler for the rest of the program.

You will be prompted to enter information for the new row that you want to insert into the `users` table in the `ecommerce` database. You will be prompted to enter the information for the new row: the email address, password, and address of delivery. We will create an SQL `INSERT` statement comprising this information (email address, password, and address of delivery), which is supposed to be entered by users. Thereafter, we will invoke the `mysql_query` function and pass the MySQL object `conn` and the SQL `INSERT` statements to it to execute the SQL statement and insert a new row into the `users` table.

If any error occurs while executing the `mysql_query` function, an error message will be displayed on the screen and the program will terminate. If the new row is successfully inserted into the `users` table, the message `Row is inserted successfully in users table` will be displayed on the screen. Finally, we will invoke the `mysql_close` function and pass the connection handler `conn` to it to close the connection handler.

Let's open the Cygwin Terminal. We will require two Terminal windows; on one window, we will run SQL commands and on the other, we will compile and run C. Open another Terminal window by pressing *Alt+F2*. In the first Terminal window, invoke the MySQL command line by using the following command:

```cpp
$ mysql -u root -p -h 127.0.0.1
Enter password:
Welcome to the MariaDB monitor.  Commands end with ; or \g. 
Your MySQL connection id is 27 
Server version: 5.7.14-log MySQL Community Server (GPL) 
Copyright (c) 2000, 2017, Oracle, MariaDB Corporation Ab and others. 
Type 'help;' or '\h' for help. Type '\c' to clear the current input statement. 
```

To work with our `ecommerce` database, we need to make it the current database. So, open the `ecommerce` database by using the following command:

```cpp
MySQL [(none)]> use ecommerce;
Reading table information for completion of table and column names
You can turn off this feature to get a quicker startup with -A 
Database changed          
```

Now, `ecommerce` is our current database; that is, whatever SQL commands we will execute will be applied to the `ecommerce` database only. Let's use the following SQL `SELECT` command to see the existing rows in the `users` database table:

```cpp
MySQL [ecommerce]> select * from users;
Empty set (0.00 sec)  
```

The given output confirms that the `users` table is currently empty. To compile the C program, switch to the second Terminal window. Let's use GCC to compile the `adduser.c` program, as shown here:

```cpp
$ gcc adduser.c -o adduser -I/usr/local/include/mysql -L/usr/local/lib/mysql -lmysqlclient        
```

If you get no errors or warnings, that means the `adduser.c` program has compiled into an executable file, `adduser.exe`. Let's run this executable file:

```cpp
$./adduser 
Enter email address: bmharwani@yahoo.com 
Enter password: gold 
Enter address of delivery: 11 Hill View Street, New York, USA
Row is inserted successfully in users table 
```

The given C program output confirms that the new row has been successfully added to the `users` database table. To confirm this, switch to the Terminal window where the MySQL command line is open and use the following command:

```cpp
MySQL [ecommerce]> select * from users;
+---------------------+----------+------------------------------------+
| email_address       | password | address_of_delivery                |
+---------------------+----------+------------------------------------+
| bmharwani@yahoo.com | gold     | 11 Hill View Street, New York, USA | 
+---------------------+----------+------------------------------------+ 
1 row in set (0.00 sec)   
```

*Voila*! The given output confirms that the new row that was entered through C has been successfully inserted into the `users` database table.

Now, let's move on to the next recipe!                                                     

# Searching for the desired information in the database

In this recipe, we will learn how to search for information in a database table. Again, we assume that a `users` table comprising three columns, `email_address`, `password`, and `address_of_delivery`, already exists (please see the section, *Creating a MySQL database and tables*, of this chapter, where we created an `ecommerce` database and a `users` table in it). On entering an email address, the recipe will search the entire `users` database table for it, and if any row is found that matches the supplied email address, that user's password and address of delivery will be displayed on the screen.

# How to do it…

1.  Initialize a MYSQL object:

```cpp
mysql_init(NULL);
```

2.  Establish a connection to the MySQL server running at the specified host. Also, establish a connection to the `ecommerce` database:

```cpp
mysql_real_connect(conn, server, user, password, database, 0, NULL, 0)
```

3.  Enter the email address of the user whose details you want to search for:

```cpp
printf("Enter email address to search: ");
scanf("%s", emailaddress);
```

4.  Create an SQL `SELECT` statement that searches the row in the `users` table that matches the email address that was entered by the user:

```cpp
strcpy(sqlquery,"SELECT * FROM users where email_address like \'");
strcat(sqlquery,emailaddress);
strcat(sqlquery,"\'");
```

5.  Execute the SQL `SELECT` statement. Terminate the program if the SQL query does not execute or some error occurs:

```cpp
if (mysql_query(conn, sqlquery) != 0)                                 
{                                                                                                                                fprintf(stderr, "No row found in the users table with this email     address\n");                                                             
    exit(1);                                                                                     }  
```

6.  If the SQL query executes successfully then the row(s) that matches the specified email address are retrieved and assigned to a `resultset`:

```cpp
resultset = mysql_use_result(conn);
```

7.  Use a `while` loop to extract one row at a time from the `resultset` and assign it to the array row:

```cpp
while ((row = mysql_fetch_row(resultset)) != NULL)
```

8.  The information of the entire row is shown by displaying the subscripts `row[0]`, `row[1]`, and `row[2]`, respectively:

```cpp
printf("Email Address: %s \n", row[0]);
printf("Password: %s \n", row[1]);
printf("Address of delivery: %s \n", row[2]);
```

9.  Memory that's allocated to the `resultset` is freed up:

```cpp
mysql_free_result(resultset);
```

10.  The opened connection handler is closed:

```cpp
mysql_close(conn);
```

The `searchuser.c` program for searching in a specific row in a MySQL database table is shown in the following code:

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

Now, let's go behind the scenes to understand the code better.

# How it works...

We will start by invoking the `mysql_init` function to initialize a `MYSQL` object by the name `conn`. Thereafter, we will invoke the `mysql_real_connect` function and pass the `MYSQL` object `conn` to it along with the valid user ID, password, and the host details. The `mysql_real_connect` function will establish a connection to the MySQL server running at the specified host and will also connect to the supplied database, `ecommerce`. The `MYSQL` object `conn` will act as the connection handler for the rest of the program. Wherever a connection to the MySQL server and `ecommerce` database is required, referring to `conn` will suffice.

If any error occurs in establishing a connection to the MySQL database engine or the `ecommerce` database, an error message will be displayed and the program will terminate. If a connection to the MySQL database engine is established successfully, you will be prompted to enter the email address of the user whose details you want to search for.

We will create an SQL `SELECT` statement that will search the row in the `users` table that matches the email address entered by the user. Then, we will invoke the `mysql_query` function and pass the created SQL `SELECT` statement to it, along with the connection handler `conn`. If the SQL query does not execute or some error occurs, the program will terminate after displaying an error message. If the query is successful, then the resulting row(s) that satisfy the condition (that is, the row(s) that match the supplied email address) will be retrieved by invoking the `mysql_use_result` function and will be assigned to the result set, `resultset`.

We will then invoke the `mysql _fetch_row` function in a `while` loop that will extract one row at a time from the `resultset`; that is, the first row from the `resultset` will be accessed and assigned to the array row.

Recall that the `users` table contains the following columns:

*   `email_address varchar(30)`
*   `password varchar(30)   `    
*   `address_of_delivery text`

Consequently, the array row will contain complete information of the accessed row, where the subscript `row[0]` will contain the data of the `email_ address` column, `row[1]` will contain the data of the column password, and `row[2]` will contain the data of the `address_of_delivery` column. The information of the entire row will be displayed by displaying the subscripts `row[0]`, `row[1]`, and `row[2]`, respectively.

At the end, we will invoke the `mysql_free_result` function to free up the memory that was allocated to the `resultset`. Then, we will invoke the `mysql_close` function to close the opened connection handler `conn`.

Let's open the Cygwin Terminal. We will require two Terminal windows; on one window, we will run SQL commands and on the other, we will compile and run C. Open another Terminal window by pressing *Alt+F2*. In the first Terminal window, invoke the MySQL command line by using the following command:

```cpp
$ mysql -u root -p -h 127.0.0.1 
Enter password: 
Welcome to the MariaDB monitor.  Commands end with ; or \g. 
Your MySQL connection id is 27 
Server version: 5.7.14-log MySQL Community Server (GPL) 
Copyright (c) 2000, 2017, Oracle, MariaDB Corporation Ab and others. 
Type 'help;' or '\h' for help. Type '\c' to clear the current input statement. 
```

To work with our `ecommerce` database, we need to make it the current database. So, open the `ecommerce` database by using the following command:

```cpp
MySQL [(none)]> use ecommerce; 
Reading table information for completion of table and column names 
You can turn off this feature to get a quicker startup with -A 
Database changed           
```

Now, `ecommerce` is our current database; that is, whatever SQL commands we will execute will be applied to the `ecommerce` database only. Let's use the following SQL `SELECT` command to see the existing rows in the `users` database table:

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

The given output shows that there are three rows in the `users` table.

To compile the C program, switch to the second Terminal window. Let's use GCC to compile the `searchuser.c` program, as shown here:

```cpp
$ gcc searchuser.c -o searchuser -I/usr/local/include/mysql -L/usr/local/lib/mysql -lmysqlclient         
```

If you get no errors or warnings, that means the `searchuser.c` program has compiled into an executable file, `searchuser.exe`. Let's run this executable file:     

```cpp
$ ./searchuser 
Enter email address to search: bmharwani@yahoo.com 
The details of the user with this email address are as follows: 
Email Address:bmharwani@yahoo.com
Password: gold 
Address of delivery: 11 Hill View Street, New York, USA 
```

*Voila*! We can see that complete information of the user with their email address, [bmharwani@yahoo.com](mailto:bmharwani@yahoo.com), is displayed on the screen.

Now, let's move on to the next recipe!

# Updating information in the database

In this recipe, we will learn how to update information in a database table. We assume that a `users` database table already exists, comprising of three columns—`email_address`, `password`, and `address_of_delivery` (please see the beginning of this chapter, where we learned to create a database and a table in it). On entering an email address, all the current information of the user (that is, their password and address of delivery) will be displayed. Thereafter, the user will be prompted to enter a new password and address of delivery. This new information will be updated against the current information in the table.

# How to do it…

1.  Initialize a `MYSQL` object:

```cpp
mysql_init(NULL);
```

2.  Establish a connection to the MySQL server running at the specified host. Also, generate a connection handler. The program will terminate if some error occurs in establishing the connection to the MySQL server engine or to the `ecommerce` database:

```cpp
 if (!mysql_real_connect(conn, server, user, password, database, 0, NULL, 0)) 
 {
      fprintf(stderr, "%s\n", mysql_error(conn));
      exit(1);
 }
```

3.  Enter the email address of the user whose information has to be updated:

```cpp
printf("Enter email address of the user to update: ");
scanf("%s", emailaddress);
```

4.  Create an SQL `SELECT` statement that will search the row in the `users` table that matches the email address that was entered by the user:

```cpp
 strcpy(sqlquery,"SELECT * FROM users where email_address like \'");
 strcat(sqlquery,emailaddress);
 strcat(sqlquery,"\'");
```

5.  Execute the SQL `SELECT` statement. The program will terminate if the SQL query does not execute successfully or some other error occurs:

```cpp
if (mysql_query(conn, sqlquery) != 0) 
{ 
     fprintf(stderr, "No row found in the users table with this          email address\n"); 
     exit(1); 
 }  
```

6.  If the SQL query executes successfully, then the row(s) that match the supplied email address will be retrieved and assigned to the `resultset`:

```cpp
 resultset = mysql_store_result(conn);
```

7.  Check if there is at least one row in the `resultset`:

```cpp
if(mysql_num_rows(resultset) >0)
```

8.  If there is no row in the `resultset`, then display the message that no row was found in the `users` table with the specified email address and exit from the program:

```cpp
printf("No user found with this email address\n");
```

9.  If there is any row in the `resultset`, then access it and assign it to the array row:

```cpp
row = mysql_fetch_row(resultset)
```

10.  Information about the user (that is, the email address, password, and address of delivery, which are assigned to the subscripts `row[0]`, `row[1]`, and `row[2]`, respectively) are displayed on the screen:

```cpp
printf("Email Address: %s \n", row[0]);
printf("Password: %s \n", row[1]);
printf("Address of delivery: %s \n", row[2]);
```

11.  The memory allocated to the `resultset` is freed:

```cpp
mysql_free_result(resultset);
```

12.  Enter the new updated information of the user; that is, the new password and the new address of delivery:

```cpp
printf("Enter new password: ");
scanf("%s", upassword);
printf("Enter new address of delivery: ");
getchar();
gets(deliveryaddress);
```

13.  An SQL `UPDATE` statement is prepared that contains the information of the newly entered password and address of delivery:

```cpp
strcpy(sqlquery,"UPDATE users set password=\'");
strcat(sqlquery,upassword);
strcat(sqlquery,"\', address_of_delivery=\'");
strcat(sqlquery,deliveryaddress);
strcat(sqlquery,"\' where email_address like \'");
strcat(sqlquery,emailaddress);
strcat(sqlquery,"\'");
```

14.  Execute the SQL `UPDATE` statement. If any error occurs in executing the SQL `UPDATE` query, the program will terminate:

```cpp
if (mysql_query(conn, sqlquery) != 0)                 
{                                                                                                                                                  fprintf(stderr, "The desired row in users table could not be 
    updated\n");  
    exit(1);
 }  
```

15.  If the SQL `UPDATE` statement executes successfully, display a message on the screen informing that the user's information has been updated successfully:

```cpp
printf("The information of user is updated successfully in users table\n");
```

16.  Close the opened connection handler:

```cpp
mysql_close(conn);
```

The `updateuser.c` program for updating a specific row of a MySQL database table with new content is shown in the following code:

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

Now, let's go behind the scenes to understand the code better.

# How it works...

In this program, we first ask the user to enter the email address they want to update. Then, we search the `users` table to see if there is any row with the matching email address. If we find it, we display the current information of the user; that is, the current email address, password, and address of delivery. Thereafter, we ask the user to enter a new password and new address of delivery. The new password and address of deliver will replace the old password and address of delivery, thereby updating the `users` table.

We will start by invoking the `mysql_init` function to initialize a `MYSQL` object by the name `conn`. Then, we will pass the `MYSQL` object `conn` to the `mysql_real_connect` function that we invoked to establish a connection to the  MySQL server running at the specified host. Several other parameters will also be passed to the `mysql_real_connection` function, including a valid user ID, password, host details, and the database with which we want to work. The `mysql_real_connect` function will establish the connection to the  MySQL server running at the specified host and will declare the `MYSQL` object `conn` as the connection handler. This means that `conn` can connect to the `MySQL` server and the `ecommerce` database wherever it is used.

The program will terminate after displaying an error message if some error occurs while establishing the connection to the MySQL server engine or to the `ecommerce` database. If the connection to the MySQL database engine is established successfully, you will be prompted to enter the email address of the user whose record you want to update.

As we mentioned earlier, we will first display the current information of the user. So, we will create an SQL `SELECT` statement and we will search the row in the `users` table that matches the email address that's entered by the user. Then, we will invoke the `mysql_query` function and pass the created SQL `SELECT` statement to it, along with the connection handler `conn`.

Again, the program will terminate after displaying an error message if the SQL query does not execute successfully or some other error occurs. If the query executes successfully, then the resulting row(s) (that is, the row(s) that match the supplied email address), will be retrieved by invoking the `mysql_use_result` function and will be assigned to the `resultset`.

We will then invoke the `mysql_num_rows` function to ensure that there is at least one row in the `resultset`. If there is no row in the `resultset`, this means that no row was found in the `users` table that matches the given email address. In this case, the program will terminate after informing that no row was found in the `users` table with the given email address. If there is even a single row in the `resultset`, we will invoke the `mysql _fetch_row` function on the `resultset`, which will extract one row from the `resultset` and assign it to the array row.

The `users` table contains the following three columns:

*   `email_address varchar(30)`
*   `password varchar(30)`
*   `address_of_delivery text`

The array row will contain the information of the accessed row, where the subscripts `row[0]`, `row[1]`, and `row[2]` will contain the data of the columns `email_ address`, `password`, and `address_of_delivery`, respectively. The current information of the user is displayed by displaying the information assigned to the aforementioned subscripts. Then, we will invoke the `mysql_free_result` function to free up the memory that is allocated to the `resultset`.

At this stage, the user will be asked to enter the new password and the new address of delivery. We will prepare an SQL `UPDATE` statement that contains the information of the newly entered password and address of delivery. The `mysql_query` function will be invoked and the SQL `UPDATE` statement will be passed to it, along with the connection handler `conn`.

If any error occurs in executing the SQL `UPDATE` query, again, an error message will be displayed and the program will terminate. If the SQL `UPDATE` statement executes successfully, a message informing that the user's information has been updated successfully will be displayed. Finally, we will invoke the `mysql_close` function to close the opened connection handler `conn`.

Let's open the Cygwin Terminal. We will require two Terminal windows; on one window, we will run SQL commands and on the other, we will compile and run C. Open another Terminal window by pressing *Alt+F2*. In the first Terminal window, invoke the MySQL command line by using the following command:

```cpp
$ mysql -u root -p -h 127.0.0.1 
Enter password: 
Welcome to the MariaDB monitor.  Commands end with ; or \g. 
Your MySQL connection id is 27 
Server version: 5.7.14-log MySQL Community Server (GPL) 
Copyright (c) 2000, 2017, Oracle, MariaDB Corporation Ab and others. 
Type 'help;' or '\h' for help. Type '\c' to clear the current input statement. 
```

To work with our `ecommerce` database, we need to make it the current database. So, open the `ecommerce` database by using the following command:

```cpp
MySQL [(none)]> use ecommerce; 
Reading table information for completion of table and column names 
You can turn off this feature to get a quicker startup with -A 
Database changed            
```

Now, `ecommerce` is our current database; that is, whatever SQL commands we will execute will be applied to the `ecommerce` database only. Let's use the following SQL `SELECT` command to see the existing rows in the `users` database table:

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

We can see from the preceding output that there are three rows in the `users` table. To compile the C program, switch to the second Terminal window. Let's use GCC to compile the `updateuser.c` program, as shown here:

```cpp
$ gcc updateuser.c -o updateuser -I/usr/local/include/mysql -L/usr/local/lib/mysql -lmysqlclient           
```

If you get no errors or warnings, that means the `updateuser.c` program has compiled into an executable file, `updateuser.exe`. Let's run this executable file:

```cpp
$ ./updateuser 
Enter email address of the user to update: harwanibintu@gmail.com 
No user found with this email address                     
```

Let's run the program again and enter an email address that already exists:       

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

So, we have updated the row of the user with the email address, [bmharwani@yahoo.com](mailto:bmharwani@yahoo.com). To confirm that the row has been updated in the `users` database table too, switch to the Terminal window where the MySQL command line is running and issue the following SQL `SELECT` command: 

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

*Voila*! We can see that the row of the `users` table with the email address `bmharwani@yahoo.com` has been updated and is showing the new information.

Now, let's move on to the next recipe!

# Deleting data from the database using C

In this recipe, we will learn how to delete information from a database table. We assume that a `users` table comprising three columns, `email_address`, `password`, and `address_of_delivery`, already exists (please see the beginning of this chapter, where we created an `ecommerce` database and a `users` table in it). You will be prompted to enter the email address of the user whose row has to be deleted. On entering an email address, all the information of the user will be displayed. Thereafter, you will again be asked to confirm if the displayed row should be deleted or not. After your confirmation, the row will be permanently deleted from the table.

# How to do it…

1.  Initialize a `MYSQL` object:

```cpp
mysql_init(NULL);
```

2.  Establish a connection to the MySQL server running at the specified host. Also, generate a connection handler. If any error occurs in establishing a connection to the MySQL server engine, the program will terminate:

```cpp
  if (!mysql_real_connect(conn, server, user, password, database, 0, 
    NULL, 0)) {
      fprintf(stderr, "%s\n", mysql_error(conn));
      exit(1);
  }
```

3.  If the connection to the MySQL database engine is established successfully, you will be prompted to enter the email address of the user whose record you want to delete:

```cpp
 printf("Enter email address of the user to delete: ");
 scanf("%s", emailaddress);
```

4.  Create an SQL `SELECT` statement that will search the row from the `users` table that matches the email address that's entered by the user:

```cpp
 strcpy(sqlquery,"SELECT * FROM users where email_address like \'");
 strcat(sqlquery,emailaddress);
 strcat(sqlquery,"\'");
```

5.  Execute the SQL `SELECT` statement. The program will terminate after displaying an error message if the SQL query does not execute successfully:

```cpp
 if (mysql_query(conn, sqlquery) != 0)                 
 {                                                                                                                                   
    fprintf(stderr, "No row found in the users table with this email 
    address\n");                                                                                                     
    exit(1);                                                                     
 }  
```

6.  If the query executes successfully, then the resulting row(s) that match the supplied email address will be retrieved and assigned to the `resultset`:

```cpp
resultset = mysql_store_result(conn);
```

7.  Invoke the `mysql_num_rows` function to ensure that there is at least one row in the `resultset`:

```cpp
if(mysql_num_rows(resultset) >0)
```

8.  If there is no row in the `resultset`, that means no row was found in the `users` table that matches the given email address; hence, the program will terminate:

```cpp
printf("No user found with this email address\n");
```

9.  If there is any row in the result set, that row is extracted from the `resultset` and will be assigned to the array row:

```cpp
row = mysql_fetch_row(resultset)
```

10.  The information of the user is displayed by displaying the corresponding subscripts in the array row:

```cpp
printf("Email Address: %s \n", row[0]);
printf("Password: %s \n", row[1]);
printf("Address of delivery: %s \n", row[2]);
```

11.  The memory that's allocated to the `resultset` is freed up:

```cpp
mysql_free_result(resultset);The user is asked whether he/she really want to delete the shown record.
printf("Are you sure you want to delete this record yes/no: ");
scanf("%s", k);
```

12.  If the user enters `yes`, an SQL `DELETE` statement will be created that will delete the row from the `users` table that matches the specified email address:

```cpp
if(strcmp(k,"yes")==0)
{
    strcpy(sqlquery, "Delete from users where email_address like 
    \'");
    strcat(sqlquery,emailaddress);
    strcat(sqlquery,"\'");
```

13.  The SQL `DELETE` statement is executed. If there are any error occurs in executing the SQL `DELETE` query, the program will terminate:

```cpp
if (mysql_query(conn, sqlquery) != 0)                 
{                                                                                   
    fprintf(stderr, "The user account could not be deleted\n");                                                             
    exit(1);                                                                     
}
```

14.  If the SQL `DELETE` statement is executed successfully, a message informing that the user account with the specified email address is deleted successfully is displayed:

```cpp
printf("The user with the given email address is successfully deleted from the users table\n");
```

15.  The opened connection handler is closed:

```cpp
mysql_close(conn);
```

The `deleteuser.c` program for deleting a specific row from a MySQL database table is shown in the following code:

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

Now, let's go behind the scenes to understand the code better.

# How it works...

We will start by invoking the `mysql_init` function to initialize a `MYSQL` object by the name `conn`. We will then pass the `MYSQL` object `conn` to the `mysql_real_connect` function that we invoked to establish a connection to the  MySQL server running at the specified host. Several other parameters will also be passed to the `mysql_real_connection` function, including a valid user ID, password, host details, and the database with which we want to work. The `mysql_real_connect` function will establish a connection to the  MySQL server running at the specified host and will declare a `MYSQL` object `conn` as the connection handler. This means thar `conn` can connect to the MySQL server and the commerce database wherever it is used.

The program will terminate after displaying an error message if some error occurs while establishing a connection to the MySQL server engine or to the `ecommerce` database. If the connection to the MySQL database engine is established successfully, you will be prompted to enter the email address of the user whose record you want to delete.

We will first display the information of the user and thereafter will seek permission from the user as to whether they really want to delete that row or not. So, we will create an SQL `SELECT` statement that will search the row from the `users` table that matches the email address that was entered by the user. Then, we will invoke the `mysql_query` function and pass the created SQL `SELECT` statement to it, along with the connection handler `conn`.

Again, the program will terminate after displaying an error message if the SQL query does not execute successfully or some other error occurs. If the query executes successfully, then the resulting row(s) (that is, the row(s) that match the supplied email address) will be retrieved by invoking the `mysql_use_result` function and will be assigned to the `resultset`.

We will invoke the `mysql_num_rows` function to ensure that there is at least one row in the `resultset`. If there is no row in the `resultset`, that means no row was found in the `users` table that matches the given email address. In that case, the program will terminate after informing that no row was found in the `users` table with the given email address. If there is even a single row in the `resultset`, we will invoke the `mysql _fetch_row` function on the `resultset`, which will extract one row from the `resultset` and assign it to the array row.

The `users` table contains the following three columns:

*   `email_address varchar(30)`
*   `password varchar(30)`
*   `address_of_delivery text`

The array row will contain information of the accessed row, where the subscripts `row[0]`, `row[1]`, and `row[2]` will contain the data of the columns `email_ address`, `password`, and `address_of_delivery`, respectively. The current information of the user will be displayed by displaying the current email address, password, and address of delivery that's assigned to the subscripts `row[0]`, `row[1]`, and `row[2]`. Then, we will invoke the `mysql_free_result` function to free up the memory that is allocated to the `resultset`.

At this stage, the user will be asked to confirm whether they really want to delete the shown record. The user is supposed to enter `yes`, all in lowercase, to delete the record. If the user enters `yes`, an SQL `DELETE` statement will be created that will delete the row from the `users` table that matches the specified email address. The `mysql_query` function will be invoked and the SQL `DELETE` statement will be passed to it, along with the connection handler `conn`.

If any error occurs in executing the SQL `DELETE` query, again an error message will be displayed and the program will terminate. If the SQL `DELETE` statement executes successfully, a message informing that the user account with the specified mail address has been deleted successfully is displayed. Finally, we will invoke the `mysql_close` function to close the opened connection handler `conn`.

Let's open the Cygwin Terminal. We will require two Terminal windows; on one window, we will run MySQL commands and on the other, we will compile and run C. Open another Terminal window by pressing *Alt+F2*. In the first Terminal window, invoke the MySQL command line by giving the following command:

```cpp
$ mysql -u root -p -h 127.0.0.1 
Enter password: 
Welcome to the MariaDB monitor.  Commands end with ; or \g. 
Your MySQL connection id is 27 
Server version: 5.7.14-log MySQL Community Server (GPL) 
Copyright (c) 2000, 2017, Oracle, MariaDB Corporation Ab and others. 
Type 'help;' or '\h' for help. Type '\c' to clear the current input statement. 
```

To work with our `ecommerce` database, we need to make it the current database. So, open the `ecommerce` database by using the following command:

```cpp
MySQL [(none)]> use ecommerce; 
Reading table information for completion of table and column names 
You can turn off this feature to get a quicker startup with -A 
Database changed            
```

Now, `ecommerce` is our current database; that is, whatever SQL commands we will execute will be applied to the `ecommerce` database only. Let's use the following SQL `SELECT` command to see the existing rows in the `users` database table:

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

From the preceding output, we can see that there are three rows in the `users` table. To compile the C program, switch to the second Terminal window. Let's use GCC to compile the `deleteuser.c` program, as shown here:

```cpp
$ gcc deleteuser.c -o deleteuser -I/usr/local/include/mysql -L/usr/local/lib/mysql -lmysqlclient
```

If you get no errors or warnings, that means the `deleteuser.c` program has compiled into an executable file, `deleteuser.exe`. Let's run this executable file:

```cpp
$ ./deleteuser
Enter email address of the user to delete: harwanibintu@gmail.com 
No user found with this email address                
```

Now, let's run the program again with a valid email address:

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

So, the row of the user with the email address `bmharwani@yahoo.com` will be deleted from the `users` table. To confirm that the row has been deleted from the `users` database table too, switch to the Terminal window where the MySQL command line is running and issue the following SQL `SELECT` command:      

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

*Voila*! We can see that now there are only two rows left in the `users` table, confirming that one row has been deleted from the `users` table.