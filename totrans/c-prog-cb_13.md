# Appendix B

I am using Cygwin for compiling and running my C programs in this book. Cygwin is an excellent tool that provides an environment to compile and run UNIX or Linux applications on a Windows operating system. In this section of the book, we will learn how to install Cygwin.

# Installing Cygwin

To download Cygwin, visit [https://www.cygwin.com/](https://www.cygwin.com/). You will be provided with two setup files, `setup-x86_64.exe` for 64-bit installation and `setup-x86.exe` for 32-bit installation. Depending on the Windows operating system version that is installed on your machine, you can download a 64-bit or 32-bit setup file. Because I have 64-bit Windows 10 installed on my computer, I have downloaded the `setup-x86_64.exe` file. Perform the following steps:

1.  To begin the Cygwin installation, simply double-click the downloaded `setup-x86_64.exe` file. You will then see the following dialog, displaying a small introduction of the Cygwin setup program. Simply click the Next button to continue:

![](img/667f4ef7-5a9d-4842-81db-61c9b835fbc5.png)

2.  You will then see a dialog asking you to choose the type of installation you want from the following three options:

*   Install from Internet: This option will download the Cygwin files from the net and will install them. The downloaded files will be saved on the local hard disk for future use.
*   Download Without Installing: This option will download the entire Cygwin program, but will not install it (but you can install it from the downloaded files anytime in the future).
*   Installation from Local Directory: This option installs Cygwin from the setup files that were downloaded earlier.

Because we want to download and install files from the internet, let's select the first option, Install from Internet, and click the Next button, as shown in the following screenshot:

![](img/3f3c2cb2-bd9f-489e-a9bb-5f7664cddaeb.png)

3.  You will be prompted to specify the drive and directory that you want to install Cygwin in. I want to install Cygwin in the D: drive, in the folder named `cygwin64` (as shown in the following screenshot). You can specify any drive or folder you desire. You also need to specify whether you want Cygwin to be used only by you or by all of the users of this system. Because we want Cygwin to be used by all of the users of the system, we will choose the first option, selecting All Users (which is the default choice), and click the Next button, as shown in the following screenshot:

![](img/ae345bbf-0998-4b21-9370-cf8ae0daf8ca.png)

4.  You will be prompted to specify the drive and directory in which you want to store the installation files (as shown in the following screenshot). After specifying the desired drive and directory, click the Next button to continue:

![](img/e325a30d-48a5-4599-aab7-2f1d6dacf34d.png)

5.  You will be asked to specify how you are connected to the internet—that is, whether you are directly connected to the internet or are connected via a proxy server. If you are connected through a proxy, you need to specify the details of the proxy host and its port number. Because I am directly connected to the internet, I will select the Direct Connection option (as shown in the following screenshot) and then click the Next button:

![](img/e1c1e377-d7ef-4004-a52a-e26160affed2.png)

6.  You will be shown a list of sites that you can download and install Cygwin from, as shown in the following screenshot. Select any option from the list and click the Next button:

![](img/b7d9ef19-9a95-492a-b488-078bc9299b11.png)

7.  You will be shown a list of categories of packages that you want to install (as shown in the following screenshot). Besides the packages that are selected by default, we need to install some files from the Database package category too. We will be requiring Database packages because we will be accessing MySQL databases and tables using C programs in this book. You can click the + symbol in the Database category node to expand it:

![](img/42888f2f-7102-4e28-a515-36c36096ecb9.png)

8.  You will get the list of the packages that are available to download under the Database category node. Select the packages that begin with the prefix mysql, such as mysql client apps, mysql bench, mysql common, mysql server, as shown in the following screenshot. After selecting the required packages, click the Next button to download and install Cygwin:

![](img/91657b55-959d-4dc8-aa47-c2e1890a8356.png)

9.  The Cygwin files will then be downloaded from the net and installed on your machine. Once Cygwin is successfully installed, you will get a confirmation dialog. Click the Finish button to close the dialog box:

![](img/0f9744c0-3653-4ef2-a5e8-9437cdc138e7.png)

Now you are ready to use Cygwin for implementing the recipes in this book.