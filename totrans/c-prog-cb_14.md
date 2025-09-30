# Appendix C

In this section of the book, we will quickly review how to install MySQL Server. This is required to implement the recipes in [Chapter 8](afdb55d7-2322-4dce-ad9d-ff737f8c3b4b.xhtml), *Using MySQL Database*.

# Installing MySQL Server

You need to download the MySQL Community Server from, [https://dev.mysql.com/downloads/](https://dev.mysql.com/downloads/). The latest version of MySQL Server that is available at the time of writing is 8.0.15\. The installer file that will be downloaded is named `mysql-installer-community-8.0.15.0.msi`.

1.  Simply double-click the file to initiate the installation. The first screen that will appear is a License Agreement. Go through the different terms and conditions that are mentioned. If you agree with the terms, click the, I accept the license terms checkbox and then click the Next button.
2.  The next dialog will prompt you to choose the setup type. The following options will be displayed:

*   Developer Default: It will install MySQL Server and other tools that enable developing applications too.
*   Server only: It will install only MySQL Server.
*   Client only: It will install MySQL applications and connectors, enabling access to the MySQL database tables from the client machine.
*   Full: It will install all the available MySQL products.
*   Custom: It will prompt you to select the MySQL products that you want to install.

Because we want to develop applications that access the MySQL database tables, we will select the Developer Default option and click the Next button, as shown in the following screenshot:

![](img/814fb325-bc0a-4e10-bc8f-d46e5e0be830.png)

3.  You will be prompted to specify the directory where the MySQL Server files need to be installed. Also, you will be asked to specify the data directory. The dialog box also shows the directories by default. You can go ahead with the default directories for installing MySQL Server and the data files, then click the Next button to continue.

4.  Simply accept the default values in the dialog boxes and click the Next button to continue with the installation procedure. You will be shown a dialog box indicating the products that will be installed on your machine. Click the Execute button to download and install the shown products.
5.  When the products are downloaded and installed, you will be shown a dialog box showing the list of products that are installed on your machine, as shown in the following screenshot. Click the Next button to move further:

![](img/547ab369-61d7-431d-bd48-20bfa43cc267.png)

6.  The next dialog will show you the following things that need to be configured on your machine:

*   MySQL Server
*   MySQL Router
*   Samples and Examples

Click the Next button to configure these three items, as shown in the following screenshot:

![](img/e35230f3-dc77-42db-9984-e60a62a491af.png)

7.  You will be asked whether you want to configure MySQL Server as a standalone server or as a Sandbox InnoDB Cluster setup. In a Sandbox InnoDB Cluster setup, you can easily configure and manage at least three MySQL Server instances in the format of InnoDB clusters. Each of the MySQL Servers has the capability to replicate data within InnoDB clusters. The cluster is smart enough to reconfigure automatically in case a failure occurs in any server instance. But because we want a single standalone server instance, select the Standalone MySQL Server option and click the Next button to continue, as shown in the following screenshot:

![](img/d5f971f8-742e-417c-ba76-db3b092815f3.png)

8.  You will be prompted to select the server configuration type. You need to select the configuration type that suits your needs and the available resources. The following three options will be displayed:

*   Development Computer: In this configuration type, your machine can run MySQL Server along with other servers and different development frameworks. The machine will be used to develop applications that can access the MySQL database and tables. The MySQL Server will be configured to use the smallest amount of memory.
*   Server: In this configuration type, besides MySQL Server, other servers might also be run, including web servers. MySQL Server will be configured to use a medium amount of memory.
*   Dedicated: In this setup type, the machine will be dedicated to run only MySQL Server. Because no other applications or servers will be running on this machine, MySQL Server will be configured to use the majority of the available memory.

Because we will be using this machine for development, that is, we will be running MySQL Server along with other servers and applications, we will choose the default Config Type, Development Computer. 

9.  The dialog box also asks you how you want your applications to connect to MySQL Server. By default, TCP/IP networking is selected, along with port `3306`. Also, a checkbox is selected by default that opens the Windows firewall ports for network access. Besides this, you will be provided with two more options, Named Pipe and Shared Memory to connect with MySQL Server.

If you want to connect with MySQL Server through a named pipe, you need to enable and define the Pipe Name. Similarly, if you want to connect with MySQL Server using the Shared Memory, you need to enable and then define the Memory Name.

We will keep the default connectivity options TCP/IP, along with port `3306`. Click the Next button to continue, as shown in the following screenshot:

![](img/b6c41347-67a6-4d1d-9c7f-ee9816faa9de.png)

10.  The next dialog prompts you to select the authentication method you want to use to connect with the MySQL Server. The following two options will be displayed:

*   Use Strong Password Encryption for Authentication (RECOMMENDED)
*   Use Legacy Authentication Method (Retain MySQL 5.x Compatibility)

Choose the Use Legacy Authentication Method option for any application that cannot be updated to use the latest MySQL 8.0 connectors and drivers. Hence, keep the default first option, Use Strong Password Encryption for Authentication, selected and click the Next button to continue, as shown in the following screenshot:

![](img/34651c04-cd14-4388-9d3a-7b59549eada0.png)

11.  In the next dialog, you will be prompted to enter a Root Account Password. Enter a strong password comprising lower and upper case letters, numbers, and symbols. After entering the Current Root Password, click the Check button on its right to confirm whether the entered password is strong or not. If you get a tick on clicking the Check button, that means the entered password is perfectly fine. Also, please remember the password as you will be required to enter it in the future:

![](img/ef3d7d91-1bcc-4fbc-ad28-9a616bed4fbd.png)

12.  In the next dialog, you will be asked to Configure MySQL Server as a Windows Service. If MySQL Server is configured as a Windows service, it will automatically start and stop when Windows starts and stops. By default, a Windows Service Name will be displayed based on the MySQL version that you are using. Also, a checkbox, Start the MySQL Server at System Startup, will be selected by default. We will stick to the default values. Also, you will be asked whether you want to run the Windows service under a Standard System Account or under a Custom User. Standard System Account will be found auto selected by default. Because we want MySQL Server to run as a Windows service under a Standard System Account, we will keep the default options selected and click the Next button to move on, as shown in the following screenshot:

![](img/5dfa07e2-1ac9-4db0-b6e5-b085756fb69d.png)

13.  The next screen will ask whether to apply the configuration settings that have been chosen by you. Click the Execute button at the bottom of the dialog to apply the configuration settings selected by us. On application of the configuration steps, you will get the following dialog box informing us that MySQL Server has been configured successfully. Click the Finish button to continue:

![](img/d84cc77e-90e7-4581-b0c0-c3f338d3735a.png)

14.  You will be informed that now MySQL Router needs to be configured, and you will be asked to select the Next button to proceed with the configuration of MySQL Router. Click the Next button to display the MySQL Router configuration dialog. The router is used to route the heavy database traffic to the running MySQL Servers so that resources are used efficiently and the database tables are accessed quickly and efficiently.
15.  Because we are using a standalone installation of MySQL Server, we do not need to configure MySQL Router, so, click the Finish button to close the router configuration dialog.

16.  The next dialog will ask whether you want to configure samples and examples for MySQL Server. The sample schema can be directly used in our applications, and examples help us understand MySQL better. So, click the Next button to configure samples and examples. You get a dialog box that prompts you to select the server on which the sample schemas and data needs to be created (see the following screenshot). Because we have installed a standalone MySQL server, only one MySQL server instance will be displayed, and that too will be selected by default.

At the bottom of the dialog, you will be prompted to enter the root password to confirm authentication before creating the samples and examples. After entering the root password, click on the Check button to confirm authentication. If you get a tick with a message saying All connections succeeded, that means you have entered the root password correctly and you can go ahead and click the Next button to begin the procedure of creating of samples and examples, as shown in the following screenshot:

![](img/455f5843-4bc7-46ea-82fb-493b2946359e.png)

17.  The next dialog will prompt you to click the Execute button to run the scripts that create sample schemas and related data. Click the Execute button to begin the procedure. When the script is executed and the samples and examples are successfully created and configured on the running MySQL server, you get the dialog box, as shown in the following screenshot. Click the Finish button to move on:

![](img/64cdd248-8044-464b-aaa6-7ed771f93855.png)

18.  The next dialog box confirms that the task of configuring MySQL Server, MySQL Router, and the samples and examples is complete. So, click the Next button to move on.

19.  The final dialog box confirms that the installation of MySQL Server and its related products is complete (see the following screenshot). The two checkboxes, Start MySQL Workbench after Setup and Start MySQL Shell after Setup, will be auto selected by default. You can keep the two checkboxes selected if you want to start them once the setup is complete, or you can uncheck either of the checkboxes if you wish to invoke them later when required. Click on the Finish button to finish the installation:

![](img/bfb7f41a-302e-4669-80a7-cefc95706130.png)

Now, you are ready to use MySQL to implement the recipes in [Chapter 8](afdb55d7-2322-4dce-ad9d-ff737f8c3b4b.xhtml), *Using MySQL Database*.