# Installation

In this chapter, you will learn how to install Qt on your development machine, including Qt Creator, an IDE tailored to use with Qt. You will see how to configure the IDE for your needs and learn the basic skills to use that environment. By the end of this chapter, you will be able to prepare your working environment for both desktop and embedded platforms using the tools included in the Qt release.

The main topics covered in this chapter are as follows:

*   Installing Qt and its developer tools
*   Main controls of Qt Creator
*   Qt documentation

# Installing the Qt SDK

Before you can start using Qt on your machine, it needs to be downloaded and installed. Qt can be installed using dedicated installers that come in two flavors: the online installer, which downloads all the needed components on the fly, and a much larger offline installer, which already contains all the required components. Using an online installer is easier for regular desktop installs, so we prefer this approach.

# Time for action – Installing Qt using an online installer

All Qt resources, including the installers, are available at [https://qt.io](https://www.qt.io/). To obtain the open source version of Qt, go to [https://www.qt.io/download-open-source/](https://www.qt.io/download-open-source/). The page suggests the online installer for your current operating system by default, as shown in the following screenshot. Click on the Download Now button to download the online installer, or click on View All Downloads to select a different download option:

![](img/06afdb67-182a-4ce2-9ce0-5d0925043af6.png)

When the download is complete run the installer, as shown:

![](img/dec2a75e-1445-4cf6-ae41-a5ea38ee752d.png)

Click on Next to begin the installation process. If you are using a proxy server, click on Settings and adjust your proxy configuration. Then, either log into your Qt Account or click on Skip, if you don't have one.

Click on Next again, and after a while of waiting as the downloader checks remote repositories, you'll be asked for the installation path. Ensure that you choose a path where you have write access and enough free space. It's best to put Qt into your personal directory, unless you ran the installer as the system administrator user. Clicking on Next again will present you with the choice of components that you wish to install, as shown in the following screenshot. You will be given different choices depending on your platform:

![](img/cecd3474-6c67-4c23-8c63-6b4e163dc805.png)

Before we continue, you need to choose which Qt version you want to install. We recommend that you use the most recent stable version, that is, the first item under the Qt section. Ignore the Preview section, as it contains prerelease packages that may be unstable. If you want to be fully consistent with the book, you can choose Qt 5.9.0, but it's not required. The installer also allows you to install multiple Qt versions at once.

Expand the section corresponding to the Qt version you want to install, and choose whichever platforms you need. Select at least one desktop platform to be able to build and run desktop applications. When in Windows, you have to make additional choices for the desktop builds. Select the 32-bit or 64-bit version and choose the compiler you want to be working with. If you have a Microsoft C++ compiler (provided with Visual Studio or Visual C++ Build Tools), you can select the build corresponding to the installed MSVC version. If you don't have a Microsoft compiler or you simply don't want to use it, choose the MinGW build and select the corresponding MinGW version in the Tools section of the package tree.

If you want to build Android applications, choose the option corresponding to the desired Android platform. In Windows, you can select a UWP build to create Universal Windows Platform applications.

The installer will always install Qt Creator—the IDE (integrated development environment) optimized for creating Qt applications. You may also select Qt add-ons that you want to use.

After choosing the required components and clicking on Next again, you will have to accept the licensing terms for Qt by marking an appropriate choice, as shown in the following screenshot:

![](img/2a56f29e-39c4-4570-8e5b-0942deb8fb11.png)

After you click on Install, the installer will begin downloading and installing the required packages. Once this is done, your Qt installation will be ready. At the end of the process, you will be given an option to launch Qt Creator:

![](img/870b2c6b-2fe3-4834-9f8b-3a885bda9983.png)

# What just happened?

The process we went through results in the whole Qt infrastructure appearing on your disk. You can examine the directory you pointed to the installer to see that it created a number of subdirectories in this directory, one for each version of Qt chosen with the installer, and another one called `Tools` that contains Qt Creator. The Qt directory also contains a `MaintenanceTool` executable, which allows you to add, remove, and update the installed components. The directory structure ensures that if you ever decide to install another version of Qt, it will not conflict with your existing installation. Furthermore, for each version, you can have a number of platform subdirectories that contain the actual Qt installations for particular platforms.

# Qt Creator

Now that Qt is installed, we will get familiar with Qt Creator and use it to verify the installation.

# Qt Creator's modes

After Qt Creator starts, you should be presented with the following screen:

![](img/d4f13116-0455-4178-b552-0f8f7dab9877.png)

The panel on the left allows you to switch between different **modes** of the IDE:

*   Welcome mode: Allows you to quickly open last sessions, projects, load examples, and tutorials.
*   Edit mode: The main mode used to edit the source code of your applications.
*   Design mode: Contains a visual form editor. Design mode is automatically activated when you create or open a Qt Widgets form file (`.ui`) or a QML form file (`.ui.qml`).
*   Debug mode: Automatically activated when you launch the application under debugger. It contains additional views for displaying the call stack, the break point list, and values of local variables. More views (such as thread lists or values of registers) can be enabled when needed.
*   Projects mode: Allows you to configure how Qt Creator will build and run your application. For example, you can choose which Qt version it will use or add command-line arguments here.
*   Help mode: Provides access to the Qt documentation. We will focus on this topic later in the chapter.

# Setting up compilers, Qt versions, and kits

Before Qt Creator can build and run projects, it needs to know which Qt builds, compilers, debuggers, and other tools are available. Fortunately, Qt installer will usually do it automatically, and Qt Creator is able to automatically detect tools that are available system-wide. However, let's verify that our environment is properly configured. From the Tools menu, choose Options. Once a dialog box pops up, choose Build & Run from the side list. This is the place where we can configure the way Qt Creator can build our projects. A complete build configuration is called a **kit**. It consists of a Qt installation and a compiler that will be executed to perform the build. You can see tabs for all the three entities in the Build & Run section of the Options dialog box.

Let's start with the Compilers tab. If your compiler was not autodetected properly and is not in the list, click on the Add button, choose your compiler type from the list, and fill the name and path to the compiler. If the settings were entered correctly, Creator will autofill all the other details. Then, you can click on Apply to save the changes.

Next, you can switch to the Qt Versions tab. Again, if your Qt installation was not detected automatically, you can click on Add. This will open a file dialog box where you will need to find your Qt installation's directory, where all the binary executables are stored (usually in the `bin` directory), and select a binary called `qmake`. Qt Creator will warn you if you choose a wrong file. Otherwise, your Qt installation and version should be detected properly. If you want, you can adjust the version name in the appropriate box.

The last tab to look at is the Kits tab. It allows you to pair a compiler with the Qt version to be used for compilation. In addition to this, for embedded and mobile platforms, you can specify a device to deploy to and a `sysroot` directory containing all the files needed to build the software for the specified embedded platform. Check that the name of each kit is descriptive enough so that you will be able to select the correct kit (or kits) for each of your applications. If needed, adjust the names of the kits.

# Time for action – Loading an example project

Examples are a great way to explore the capabilities of Qt and find the code required for some typical tasks. Each Qt version contains a large set of examples that are always up to date. Qt Creator provides an easy way to load and compile any example project.

Let's try loading one to get familiar with Qt Creator's project editing interface. Then, we will build the project to check whether the installation and configuration were done correctly.

In Qt Creator, click on the Welcome button in the top-left corner of the window to switch to the Welcome mode. Click on the Examples button (refer to the previous screenshot) to open the list of examples with a search box. Ensure that the kit that you want to use is chosen in the drop-down list next to the search box. In the box, enter `aff` to filter the list of examples and click on Affine Transformations to open the project. If you are asked whether you want to copy the project to a new folder, agree.

After selecting an example, an additional window appears that contains the documentation page of the loaded example. You can close that window when you don't need it. Switch back to the main Qt Creator window.

Qt Creator will display the Configure Project dialog with the list of available kits:

![](img/b713f18a-47aa-4bd8-b4d2-6b5904937a68.png)

Verify that the kits you want to use are marked with check boxes, and click on the Configure Project button. Qt Creator will then present you with the following window:

![](img/426493c7-c4f5-4e0c-9d93-bef5f2961b55.png)

This is the Edit mode of Qt Creator. Let's go through the most important parts of this interface:

*   **Project tree** is located at the top-left of the window. It displays all open projects and the hierarchy of files within them. You can double-click on a file to open it for editing. The context menu of projects, directories, and files in the project tree contains a lot of useful functions.

*   At the bottom-left of the window, there's a list of **open documents**. The file selected in this list will appear in the code editor in the center of the window. If the selected file is a Qt Designer form, Qt Creator will automatically switch to the Design mode. Each file in the list has a close button.
*   The Type to locate field is present at the left of the bottom panel. If you want to quickly navigate to another file in the project, type the beginning of its name in the field and select it in the pop-up list. Special prefixes can be used to enable other search modes. For example, the `c` prefix allows you to search for C++ classes. You can press *Ctrl* + *K* to activate this field.
*   The buttons at the bottom of the left panel allow you to build and run your current project under debugger, or normally. The button above them displays names of the current project and the current build configuration (for example, Debug or Release) and allows you to change them.
*   The output panes appear below the code editor when you select them in the bottom panel. The Issues pane contains compiler errors and other related messages. The Search Results pane allows you to run a text search in the entire project and view its results. The Application Output pane displays the text your application has printed to its standard output (`stderr` or `stdout`).

Qt Creator is highly configurable, so you can adjust the layout to your liking. For example, it's possible to change the locations of panes, add more panes, and change keyboard shortcuts for every action.

# Qt documentation

Qt project has very thorough documentation. For each API item (class, method, and so on), there is a section in the documentation that describes that item and mentions things that you need to know. There are also a lot of overview pages describing modules and their parts. When you are wondering what some Qt class or module is made for or how to use it, the Qt documentation is always a good source of information.

Qt Creator has an integrated documentation viewer. The most commonly used documentation feature is context help. To try it out, open the `main.cpp` file, set the text cursor inside the `QApplication` text, and press *F1*. The help section should appear to the right of the code editor. It displays the documentation page for the `QApplication` class. The same should work for any other Qt class, method, macro, and so on. You can click on the Open in Help Mode button on top of the help page to switch to the Help mode, where you have more space to view the page.

Another important feature is the search in documentation index. To do that, go to the Help mode by clicking on the Help button on the left panel. In Help mode, in the top-left corner of the window, there is a drop-down list that allows you to select the mode of the left section: Bookmarks, Contents, Index, or Search. Select Index mode, input your request in the Look for: text field and see whether there are any search results in the list below the text field. For example, try typing `qt core` to search for the Qt Core module overview. If there are results, you can press *Enter* to quickly open the first result or double-click on any result in the list to open it. If multiple Qt versions are installed, a dialog may appear where you need to select the Qt version you are interested in.

Later in this book, we will sometimes refer to Qt documentation pages by their names. You can use the method described previously to open these pages in Qt Creator.

# Time for action – Running the Affine Transformations project

Let's try building and running the project to check whether the building environment is configured properly. To build the project, click on the hammer icon (Build) at the bottom of the left panel. At the right of the bottom panel, a grey progress bar will appear to indicate the build progress. When the build finishes, the progress bar turns green if the build was successful or red otherwise. After the application was built, click on the green triangle icon to run the project.

Qt Creator can automatically save all files and build the project before running it, so you can just hit the Run (*Ctrl* + *R*) or Start Debugging (*F5*) button after making changes to the project. To verify that this feature is enabled, click on Tools and Options in the main menu, go to the Build & Run section, go to the General tab, and check that the Save all files before build, Always build project before deploying it, and Always deploy project before running it options are checked.

If everything works, after some time, the application should be launched, as shown in the next screenshot:

![](img/bb4f2200-6e99-43b3-a5f0-4a3e8f792c94.png)

# What just happened?

How exactly was the project built? To see which kit and which build configuration was used, click on the icon in the action bar directly over the green triangle icon to open the build configuration popup, as shown in the following screenshot:

![](img/a8ae3632-b9a3-41b1-9671-cbc97ec4795a.png)

The exact content that you get varies depending on your installation, but in general, on the left, you will see the list of kits configured for the project and on the right, you will see the list of build configurations defined for that kit. You can click on these lists to quickly switch to a different kit or a different build configuration. If your project is configured only for one kit, the list of kits will not appear here.

What if you want to use another kit or change how exactly the project is built? As mentioned earlier, this is done in the Projects mode. If you go to this mode by pressing the Projects button on the left panel, Qt Creator will display the current build configuration, as shown in the following screenshot:

![](img/1b0c6145-1af7-49d8-9cd7-29bc2097dbfe.png)

The left part of this window contains a list of all kits. Kits that are not configured to be used with this project are displayed in gray color. You can click on them to enable the kit for the current project. To disable a kit, choose the Disable Kit option in its context menu.

Under each enabled kit, there are two sections of the configuration. The Build section contains settings related to building the project:

*   Shadow build is a build mode that places all temporary build files in a separate build directory. This allows you to keep the source directory clean and makes your source files easier to track (especially if you use a version control system). This mode is enabled by default.
*   Build directory is the location of temporary build files (only if shadow build is enabled). Each build configuration of the project needs a separate build directory.
*   The Build steps section displays commands that will be run to perform the actual building of the project. You can edit command-line arguments of the existing steps and add custom build steps. By default, the build process consists of two steps: `qmake` (Qt's project management tool described in the previous chapter) reads the project's `.pro` file and produces a makefile, and then some variation of `make` tool (depending on the platform) reads the makefile and executes Qt's special compilers, the C++ compiler, and the linker. For more information about `qmake`, look up the `qmake Manual` in the documentation index.
*   The Build environment section allows you to view and change environment variables that will be available to the build tools.

Most variations of the `make` tool (including `mingw32-make`) accept the `-j num_cores` command-line argument that allows `make` to spawn multiple compiler processes at the same time. It is highly recommended that you set this argument, as it can drastically reduce compilation time for big projects. To do this, click on Details at the right part of the Make build step and input `-j num_cores` to the Make arguments field (replace `num_cores` with the actual number of processor cores on your system). However, MSVC `nmake` does not support this feature. To fix this issue, Qt provides a replacement tool called `jom` that supports it.

There can be multiple build configurations for each kit. By default, three configurations are generated: Debug (required for the debugger to work properly), Profile (used for profiling), and Release (the build with more optimizations and no debug information).

The Run section determines how the executable produced by your project will be started. Here, you can change your program's command-line arguments, working directory, and environment variables. You can add multiple run configurations and switch between them using the same button that allows you to choose the current kit and build configuration.

In most cases for desktop and mobile platforms, the binary release of Qt you download from the web page is sufficient for all your needs. However, for embedded systems, especially for ARM-based systems, there is no binary release available, or it is too heavy resource wise for such a lightweight system. Fortunately, Qt is an open source project, so you can always build it from sources. Qt allows you to choose the modules you want to use and has many more configuration options. For more information, look up Building Qt Sources in the documentation index.

# Summary

By now, you should be able to install Qt on your development machine. You can now use Qt Creator to browse the existing examples and learn from them or to read the Qt reference manual to gain additional knowledge. You should have a basic understanding of Qt Creator's main controls. In the next chapter, we will finally start using the framework, and you will learn how to create graphical user interfaces by implementing our very first simple game.