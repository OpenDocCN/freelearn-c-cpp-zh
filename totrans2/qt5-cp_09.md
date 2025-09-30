# Chapter 9. Deploying Applications on Other Devices

After development, it's time to distribute your application. We'll use an example application, `Internationalization`, from the previous chapter to demonstrate how to spread your Qt application to Windows, Linux, and Android. The following topics will be covered in this chapter:

*   Releasing Qt applications on Windows
*   Creating an installer
*   Packaging Qt applications on Linux
*   Deploying Qt applications on Android

# Releasing Qt applications on Windows

After the development stage, you can build your application using `release` as the build configuration. In the `release` configuration, your compiler will optimize the code and won't produce debug symbols, which in turn reduces the size. Please ensure that the project is in the `release` configuration.

Before we jump into the packaging procedure, I'd like to talk about the difference between static and dynamic linking. You have probably been using dynamic linking of Qt libraries throughout this book. This can be confirmed if you download the **Community Edition** from the Qt website.

So, what does dynamic linking mean? Well, it means that when an executable file gets executed, the operating system will load and link the necessary shared libraries at runtime. In other words, you'll see a lot of `.dll` files on Windows and `.so` files on the Unix platforms. This technique allows developers to update these shared libraries and the executable separately, which means that you don't need to rebuild the executable file if you change shared libraries, so long as their ABIs are compatible. Although this method is more flexible, developers are warned to take care to avoid **DLL Hell**.

The most commonly used solution to DLL Hell on Windows is to choose static linking instead. By contrast, static linking will resolve all the function calls and variables at compile time and copy them into the target to produce a standalone executable. The advantages are obvious. Firstly, you don't need to ship all necessary and shared libraries. There won't be DLL Hell in this situation. On Windows, static libraries may get `.lib` or `.a` as extensions depending on the compiler you use, whereas they usually get `.a` on the Unix platforms.

To make a clear comparison, a table is made for you to see the differences between the dynamic and static linking:

|   | Dynamic Linking | Static Linking |
| --- | --- | --- |
| **Library types** | Shared libraries | Static libraries |
| **Executable size** | Considerably smaller | Greater than dynamically linked |
| **Library updates** | Only libraries themselves | Executable file needs to be rebuilt |
| **Incompatible libraries** | Need to take care to avoid this | Won't happen |

However, if the shared libraries shipped with dynamically linked executable files are counted as part of the package, the dynamic style package will be larger than the statically linked standalone executable files.

Now, back to the topic! Since there is no standard Qt runtime library installer for Windows, the best routine is to produce a statically linked target because the package to be released will be smaller, and the executable is immune to DLL Hell.

However, as mentioned previously, the Qt libraries you downloaded can only be used for dynamic linking applications because they are shared libraries. It is viable to compile Qt as static libraries. However, before you proceed, you need to know the licenses of Qt.

Currently, in addition to the Qt Open Source License, there is also the Qt Commercial License. For open source licenses, most of the Qt libraries are licensed under **The GNU Lesser General Public License** (**LPGL**). In this case, if you build your application statically linked with the Qt libraries, your application is subject to provide users the source code of your application under LGPL. Your application may stay proprietary and closed source if it's dynamically linked with the Qt libraries. In other words, if you want to link an application statically and keep it proprietary, you have to purchase the Qt commercial license. For details about Qt licensing, refer to [http://www.qt.io/licensing/](http://www.qt.io/licensing/).

If you decide to use static linking, you might have to compile the Qt libraries statically before building your application. In this case, the executable target is the only thing that needs to be packaged and released. Don't forget the QM files if your application has multi-language support, as mentioned previously.

On the other hand, if you want to go the dynamic way, it'd need some extra effort. Firstly, there are some core DLLs that have to exist and the list is different depending on the compiler. The following table includes both MSVC and MinGW/GCC scenarios:

| MSVC 2013 | MinGW/GCC |
| --- | --- |
| `msvcp120.dll` | `libgcc_s_dw2-1.dll` |
| `msvcr120.dll` | `libstdc++-6.dll` |
|   | `libwinpthread-1.dll` |

There are common DLLs that need to be included, such as `icudt53.dll`, `icuin53.dll`, and `icuuc53.dll`. You can find these files in the Qt libraries directory. Take MinGW/GCC as an example; they're located in `QT_DIR\5.4\mingw491_32\bin` where `QT_DIR` is the Qt installation path, such as `D:\Qt`. Note that the later versions of Qt may have slightly different filenames.

Besides, there is no need to ship `msvcp120.dll` and `msvcr120.dll` if the target users have installed **Visual C++ Redistributable Packages** for Visual Studio 2013, which can be downloaded from [http://www.microsoft.com/en-ie/download/details.aspx?id=40784](http://www.microsoft.com/en-ie/download/details.aspx?id=40784).

After this, you may want to check other DLLs you'll need by looking into the project file. Take the `Internationalization` project as an example. Its project file, `Internationalization.pro`, gives us a clue. There are two lines related to the QT configuration, shown as follows:

[PRE0]

The `QT` variable includes the `core gui` widgets. In fact, all the Qt applications will include `core` at least, while others are dependent. In this case, we have to ship `Qt5Core.dll`, `Qt5Gui.dll`, and `Qt5Widgets.dll` along with the executable target.

Now, build the `Internationalization` project with MinGW/GCC. The executable target, `Internationalization.exe`, should be located inside the `release` folder of the build directory, which can be read from the **Projects** mode. Next, we create a new folder named `package` and copy the executable file there. Then, we copy the needed DLLs to `package` as well. Now, this folder should have all the necessary DLLs as shown here:

![Releasing Qt applications on Windows](img/4615OS_09_01.jpg)

In most cases, if a required library is missing, the application won't run while the operating system will prompt the missing library name. For instance, if `Qt5Widgets.dll` is missing, the following system error dialog will show up when you try to run `Internationalizationi.exe`:

![Releasing Qt applications on Windows](img/4615OS_09_02.jpg)

Basically, the routine is to copy the missing libraries to the same folder that the application is in. Besides, you can use some tools such as `Dependency Walker` to get the library dependencies.

Please don't use DLLs from the `Qt Editor` folder. This version is often different from Qt Libraries you've used. In addition to these libraries, you may have to include all the resources that your application is going to use. For example, the QM files used for translation, that is, to copy the `Internationalization_de.qm` file in order to load the German translation.

The file list is as follows:

*   `icudt53.dll`
*   `icuin53.dll`
*   `icuuc53.dll`
*   `Internationalization.exe`
*   `Internationalization_de.qm`
*   `libgcc_s_dw2-1.dll`
*   `libstdc++-6.dll`
*   `libwinpthread-1.dll`
*   `Qt5Core.dll`
*   `Qt5Gui.dll`
*   `Qt5Widgets.dll`

Don't forget, this is the case for MinGW/GCC in Qt 5.4.0, while different versions and compilers might have a slightly different list, as we discussed before.

After this first-time preparation, to some extent this list is fixed. You only need to change the executable target and the QM file if it's changed. An easy way to do this is to compress all of them in `tarball`.

# Creating an installer

Although it's quick to use an archive file to distribute your application, it seems more professional if you provide users with an installer. Qt offers **Qt Installer Framework** whose latest open source version, 1.5.0 for now, can be obtained from [http://download.qt.io/official_releases/qt-installer-framework/1.5.0/](http://download.qt.io/official_releases/qt-installer-framework/1.5.0/).

For the sake of convenience, let's create a folder named `dist` under the Qt Installer Framework installation path, `D:\Qt\QtIFW-1.5.0`. This folder is used to store all the application projects that need to be packaged.

Then, create a folder named `internationalization` under `dist`. Inside `internationalization`, create two folders, `config` and `packages`.

The name of the directory inside the `packages` directory acts as a domain-like, or say Java-style, identifier. In this example, we have two packages, one is the application while the other one is a translation. Therefore, it adds to the two folders in the packages directory, `com.demo.internationalization`, and `com.demo.internationalization.translation`, respectively. There will be `meta` and `data` directories present inside each of them, so the overall directory structure is sketched as follows:

![Creating an installer](img/4615OS_09_03.jpg)

Let's edit the global configuration file, `config.xml`, which is first inside the `config` directory. You need to create one file named `config.xml`.

### Note

Always remember not to use the Windows built-in Notepad to edit this file, or in fact any file. You may either use Qt Creator or other advanced editors, such as Notepad++, to edit it. This is simply because Notepad lacks of a lot of features as a code editor.

In this example, the `config.xml` file's content is pasted here:

[PRE1]

For a minimum `config.xml` file, the elements `<Name>` and `<Version>` must exist in `<Installer>`. All other elements are optional, but you should specify them if there is a need. Meanwhile, `<TargetDir>` and `<AdminTargetDir>` may be a bit confusing. They both specify the default installation path, where `<AdminTargetDir>` is to specify the installation path when it gained administrative rights. The other elements are pretty much self-explanatory. There are other elements that you can set to customize the installer. For more details, refer to [http://doc.qt.io/qtinstallerframework/ifw-globalconfig.html](http://doc.qt.io/qtinstallerframework/ifw-globalconfig.html).

Let's navigate into the `meta` folder inside `com.demo.internationalization`. This directory contains the files that specify the settings for deployment and installation. All the files in this directory, except for licenses, won't be extracted by the installer, and neither will they be installed. There must be at least a package information file, such as `package.xml`. The following example, `package.xml`, in `com.demo.internationalization/meta` is shown here:

[PRE2]

The `<Default>` element specifies whether this package should be selected by default. At the same time, we set `<ForcedInstallation>` to `true`, indicating that the end users can't deselect this package. While the `<Licenses>` element can have multiple children `<License>`, in this case we only have one. We have to provide the `license.txt` file, whose content is just a single line demonstration, as shown here:

[PRE3]

The following `package.xml` file, which is located in `com.demo.internationalization.translation/meta`, has fewer lines:

[PRE4]

The difference between `<DisplayName>` and `<Description>` is demonstrated by the following screenshot:

![Creating an installer](img/4615OS_09_04.jpg)

The `<Description>` element is the text that displays on the right-hand side when the package gets selected. It's also the text that pops up as the tooltip when the mouse hovers over the entry. You can also see the relationship between these two packages. As the name `com.demo.internationalization.translation` suggests, it is a subpackage of `com.demo.internationalization`.

The licenses will be displayed after this step and are shown in the following screenshot. If you set multiple licenses, the dialog will have a panel to view those licenses separately, similar to the one you see when you install Qt itself.

![Creating an installer](img/4615OS_09_05.jpg)

For more settings in the `package.xml` file, refer to [http://doc.qt.io/qtinstallerframework/ifw-component-description.html#package-information-file-syntax](http://doc.qt.io/qtinstallerframework/ifw-component-description.html#package-information-file-syntax).

By contrast, the `data` directories store all the files that need to be installed. In this example, we keep all files prepared previously in the `data` folder of `com.demo.internationalization`, except for the QM file. The QM file, `Internationalization_de.qm`, is kept in the `data` folder inside `com.demo.internationalization.translation`.

After all the initial preparation, we come to the final step to generate the installer application of this project. Depending on your operating system, open **Command Prompt** or **Terminal**, changing the current directory to `dist/internationalization`. In this case, it's `D:\Qt\QtIFW-1.5.0\dist\internationalization`. Then, execute the following command to generate the `internationalization_installer.exe` installer file:

[PRE5]

### Note

On Unix platforms, including Linux and Mac OS X, you'll have to use a slash (/) instead of anti-slash (\), and drop the `.exe` suffix, which makes the command slightly different, as shown here:

[PRE6]

You need to wait for a while because the `binarycreator` tool will package files in the `data` directories into the `7zip` archives, which is a time consuming process. After this, you should expect to see `internationalization_installer.exe` (or without `.exe`) in the current directory.

The installer is much more convenient, especially for a big application project that has several optional packages. Besides, it'll register and let the end users uninstall through **Control Panel**.

# Packaging Qt applications on Linux

Things are more complicated on Linux than on Windows. There are two popular package formats: **RPM Package Manager** (**RPM**) and **Debian Binary Package** (**DEB**). RPM was originally developed for **Red Hat Linux** and it's the baseline package format of **Linux Standard Base**. It's mainly used on **Fedora**, **OpenSUSE**, **Red Hat Enterprise Linux**, and its derivatives; while the latter is famous for being used in **Debian** and its well-known and popular derivative, **Ubuntu**.

In addition to these formats, there are other Linux distributions using different package formats, such as **Arch Linux** and **Gentoo**. It will take extra time to package your applications for different Linux distributions.

However, it won't be too time consuming, especially for open-source applications. If your application is open source, you can refer to the documentation to write a formatted script to compile and package your application. For details on creating an RPM package, refer to [https://fedoraproject.org/wiki/How_to_create_an_RPM_package](https://fedoraproject.org/wiki/How_to_create_an_RPM_package), whereas for DEB packaging, refer to [https://www.debian.org/doc/manuals/maint-guide/index.en.html](https://www.debian.org/doc/manuals/maint-guide/index.en.html). There is an example later that demonstrates how to package DEB.

Although it's feasible to pack proprietary applications, such as the RPM and DEB packages, they won't get into the official repository. In this case, you may want to set up a repository on your server or just release the packages via a file host.

Alternatively, you can archive your applications, similar to what we do on Windows, and write a shell script for installation and uninstallation. In this way, you can use one tarball or Qt Installer Framework to cook an installer for various distributions. But, don't ever forget to address the dependencies appropriately. The incompatible shared library issue is even worse on Linux, because almost all the libraries and applications are linked dynamically. The worst part is the incompatibility between different distributions, since they may use different library versions. Therefore, either take care of these pitfalls, or go the static linking way.

As we mentioned previously, statically linked software must be open source unless you have purchased the Qt commercial license. This dilemma makes the statically linked open source application pointless. This is not only because dynamic linking is the standard way, but also because statically linked Qt applications won't be able to use the system theme and can't benefit from system upgrades, which is not okay when security updates are involved. Anyway, you can compile your application using static linking if your application is proprietary and you get a commercial license. In this case, just like static linking on Windows, you only need to release the target executable files with the necessary resources, such as icons and translations. It's noteworthy that even if you build statically linked Qt applications, it's still impossible to run them on any Linux distributions.

Therefore, the recommended way is to install several mainstream Linux distributions on virtual machines, and then use these virtual machines to package your dynamically linked application as their own package formats. The binary package doesn't contain source code, and it's also a common practice to strip the symbols from the binary package. In this way, your source code for proprietary software won't be leaked through these packages.

We still use `Internationalization` as an example here. Let's see how to create a DEB package. The following operations were tested on the latest **Debian Wheezy**; later versions or different Linux distributions might be slightly different.

Before we package the application, we have to edit the project file, `Internationalization.pro`, to make it installable as follows:

[PRE7]

There is a concept in `qmake` called **install set**. Each install set has three members: `path`, `files`, and `extra`. The `path` member defines the destination location, while `files` tells `qmake` what files should be copied. You can specify some commands that need to be executed before other instructions in `extra`.

`TARGET` is a bit special. Firstly, it's the target executable (or library), while on the other hand, it also implies `target.files`. Therefore, we only need to specify the path of `target`. We also use the same path for `qmfile`, which includes the QM file. Don't forget to use a double dollar sign, `$$`, to use a variable. Lastly, we set the `INSTALLS` variable, which defines what is to be installed when `make install` is called. The `unix` brackets are used to limit the lines only read by `qmake` on the Unix platforms.

Now, we can get into the DEB packaging part by performing the following steps:

1.  Change your working directory (current directory) to the root of the project, that is, `~/Internationalization`.
2.  Create a new folder named `debian`.
3.  Create the four required files in the `debian` folder: `control`, `copyright`, `changelog`, and `rules`, respectively. Then, create an optional `compat` file in the `debian` folder as well.

The `control` file defines the most basic yet most critical things. This file is all about the source package and the binary package(s). The `control` file of our example is pasted here:

[PRE8]

The first paragraph is to control information for a source, whereas each of the following sets describe a binary package that the source tree builds. In other words, one source package may build several binary packages. In this case, we build only one binary package whose name is the same as `Source` and `internationalization`.

In the `Source` paragraph, `Source` and `Maintainer` are mandatory while `Section`, `Priority`, and `Standards-Version` are recommended. `Source` identifies the source package name, which can't include uppercase letters. Meanwhile, `Maintainer` contains the maintainer package's name and the e-mail address in the RFC822 format. The `Section` field specifies an application area in which the package has been classified. `Priority` is a self-explanatory field, indicating how important this package is. Lastly, `Standards-Version` describes the most recent version of the standards with which the package complies. In most cases, you should use the latest standard version, 3.9.6 for now. There are other fields that may be useful but optional. For more details, refer to [https://www.debian.org/doc/debian-policy/ch-controlfields.html](https://www.debian.org/doc/debian-policy/ch-controlfields.html).

You can specify certain packages needed for building in `Build-Depends`, similar to `qt5-qmake` and `qtbase5-dev` in our example. They're only defined for building processes and won't be included in the dependencies of binary packages.

The binary paragraphs are similar to the source except that there is no `Maintainer`, but `Architecture` and `Description` are mandatory now. For binary packages, `Architecture` can be any particular architecture or simply `any` or `all`. Specifying `any` indicates that the source package isn't dependent on any particular architecture and hence can be built on any architecture. In contrast to this, `all` means that the source package will produce only architecture-independent packages, such as documentations and scripts.

In `Depends` of the binary paragraph, we put `${shlibs:Depends}, ${misc:Depends}` instead of particular packages. The `${shlibs:Depends}` line can be used to let `dpkg-shlibdeps` generate shared library dependencies automatically. On the other hand, according to `debhepler`, you're encouraged to put `${misc:Depends}` in the field to supplement `${shlibs:Depends}`. In this way, we don't need to specify the dependencies manually, which is a relief for packagers.

The second required file, `copyright`, is to describe the licenses of the source as well as the DEB packages. In the `copyright` file, the format field is required while the others are optional. For more details about the formats of copyright, refer to [https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/](https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/). The `copyright` file in this example is shown as follows:

[PRE9]

The first paragraph is called **Header paragraph**, which is needed once and only once. The `Format` line is the only mandatory field in this paragraph, and in most cases, this line is the same. The syntax of the `Upstream-Contact` field is the same as `Maintainer` in the `control` file.

The second paragraph in this file is **Files paragraph**, which is mandatory and repeatable. In these paragraphs, `File`, `Copyright`, and `License` are required. We use an asterisk sign (`*`) indicating that this paragraph applies to all files. The `Copyright` field may contain the original statement copied from files or a shortened text. The `License` field in a `Files` paragraph describes the licensing terms for the files defined by `File`.

Following the Files paragraph, the **Stand-alone license paragraph** is optional and repeatable. We have to provide the full license text if the license is not provided by Debian. Generally speaking, only commonly-seen open-source licenses are provided. The first line must be a single license short name, which is then followed by a license text. For a license text, there must be a two space indentation in each line's head.

Don't be misled by the `changelog` filename. This file also has a special format and is used by `dpkg` to obtain the version number, revision, distribution, and urgency of your package. It's a good practice to document all the changes you have made in this file. However, you can just list the most important ones if you have a version control system. The `changelog` file in our example has the following contents:

[PRE10]

The first line is the package name, version, distribution, and urgency. The name must match the source package name. In this example, `internationalization` is the name, `1.0.0-1` is the version, `unstable` stands for the distribution, and `urgency` is `low`. Then, use an empty line to separate the first line and log entries. In the log entries, all the changes that you want to document should be listed. For each entry, there are two spaces and an asterisk sign (`*`) in the header. The last part of a paragraph is a maintainer line that begins with a space. For more details about this file and its format, refer to [https://www.debian.org/doc/debian-policy/ch-source.html#s-dpkgchangelog](https://www.debian.org/doc/debian-policy/ch-source.html#s-dpkgchangelog).

Now, we need to take a look at what `dpkg-buildpackage` will do to create the package. This process is controlled by the `rules` file; the example is pasted here:

[PRE11]

This file, similar to `Makefile`, consists of several rules. Also, each rule begins with its target declaration, while the recipes are the following lines beginning with the TAB code (not four spaces). We explicitly set Qt 5 as the Qt version, which can avoid some issues when Qt 5 coexists with Qt 4\. The percentage sign (`%`) is a special target and means any targets, which just calls the `dh` program with the target name, while `dh` is just a wrapper script, which runs appropriate programs depending on its argument, the real target.

The rest of the lines are customizations for the `dh` command. For instance, `dh_auto_configure` will call `./configure` by default. In our case, we use `qmake` to generate `Makefile` instead of a configure script. Therefore, we override `dh_auto_configure` by adding the `override_dh_auto_configure` target with `qmake` as the recipe.

Although the `compat` file is optional, you'll get bombarded with warnings if you don't specify it. Currently, you should set its content to `9`, which can be done by the following single-line command:

[PRE12]

We can generate the binary DEB package now. The `-uc` argument stands for uncheck while `-us` stands for unsign. If you have a PKG key, you may need to sign the package so that users can trust the packages you've released. We don't need source packages, so the last argument, `-b`, indicates that only the binary packages will be built.

[PRE13]

The automatically detected dependencies can be viewed in the `debian/` file, `internationalization.substvars`. This file's contents are pasted here:

[PRE14]

As we discussed earlier, the dependencies are generated by `shlibs` and `misc`. The biggest advantage is that these generated version numbers tend to be the smallest, which means the maximum backwards compatibility. As you can see, our `Internationalization` example can run on Qt 5.0.2.

If everything goes well, you'd expect a DEB file in an upper-level directory. However, you can only build the current architecture's binary package, `amd64`. If you want to build for `i386` natively, you need to install a 32-bit x86 Debian. For cross-compilation, refer to [https://wiki.debian.org/CrossBuildPackagingGuidelines](https://wiki.debian.org/CrossBuildPackagingGuidelines) and [https://wiki.ubuntu.com/CrossBuilding](https://wiki.ubuntu.com/CrossBuilding).

Installing a local DEB file is easily done with the following single-line command:

[PRE15]

After installation, we can run our application by running `/opt/internationalization_demo/Internationalization`. It should run as expected and behave exactly the same as on Windows, as shown in the following screenshot:

![Packaging Qt applications on Linux](img/4615OS_09_06.jpg)

# Deploying Qt applications on Android

The `internationalization` application requires a QM file to be loaded correctly. On Windows and Linux, we choose to install them alongside the target executable. However, this is not always a good approach, especially on Android. The path is more complicated than the desktop operating systems. Besides, we're building a Qt application instead of the Java application. Localization is definitely different from a plain Java application, as stated in the Android documentation. Hence, we're going to bundle all the resources into the `qrc` file, which will be built into the binary target:

1.  Add a new file to project by right-clicking on the project, and then select **Add New…**.
2.  Navigate to **Qt** | **Qt Resource File** in the **New File** dialog.
3.  Name it `res` and click on **OK**; Qt Creator will redirect you to edit `res.qrc`.
4.  Navigate to **Add** | **Add Prefix** and change **Prefix** to `/`.
5.  Navigate to **Add** | **Add Files** and select the .`Internationalization_de.qm` file in the dialog.

Now, we need to edit `mainwindow.cpp` to make it load the translation file from `Resources`. We only need to change the constructor of `MainWindow` where we load the translation, as shown here:

[PRE16]

The preceding code is to specify the directory for the `QTranslator::load` function. As we mentioned in the previous chapter, `:/` indicates that it's a `qrc` path. Don't add a `qrc` prefix unless it's a `QUrl` object.

We can remove the `qmfile` install set from the project file now, because we've already bundled the QM file. In other words, after this change, you don't need to ship the QM file on Windows or Linux anymore. Edit the project file, `Internationalization.pro`, as shown in the following code:

[PRE17]

Now, switch to **Projects** mode and add the **Android** kit. Don't forget to switch the build to `release`. In **Projects** mode, you can modify how Qt Creator should build the Android APK package. There is an entry in **Build Steps** called **Build Android APK**, as shown in the following screenshot:

![Deploying Qt applications on Android](img/4615OS_09_07.jpg)

Here, you can specify the Android API level and your certificate. By default, **Qt Deployment** is set to **Bundle Qt libraries in APK**, which creates a redistributable APK file. Let's click on the **Create Templates** button to generate a manifest file, `AndroidManifest.xml`. Normally, you just click on the **Finish** button on the pop-up dialog, and then Qt Creator will redirect you back to the **Edit** mode with `AndroidManifest.xml` open in the editing area, as shown here:

![Deploying Qt applications on Android](img/4615OS_09_08.jpg)

Let's make a few changes to this manifest file by performing the following steps:

1.  Change **Package name** to `com.demo.internationalization`.
2.  Change **Minimum required SDK** to `API 14: Android 4.0, 4.0.1, 4.0.2`.
3.  Change **Target SDK** to `API 19: Android 4.4`.
4.  Save the changes.

Different API levels have an impact on compatibility and the UI; you have to decide the levels carefully. In this case, we require at least Android 4.0 to run this application, which we're going to it for Android 4.4\. Generally speaking, the higher the API level, the better the overall performance is. The `Internationalization.pro` project file is automatically changed as well.

[PRE18]

Now, build a `release` build. The APK file is created in `android-build/bin` inside the project build directory. The APK filename is `QtApp-release.apk` or `QtApp-debug.apk` if you don't set your certificate. If you're going to submit your application to Google Play or any other Android markets, you have to set your certificate and upload `QtApp-release.apk` instead of `QtApp-debug.apk`. Meanwhile, `QtApp-debug.apk` can be used on your own devices to test the functionality of your application.

The screenshot of `Internationalization` running on HTC One is shown as follows:

![Deploying Qt applications on Android](img/4615OS_09_09.jpg)

As you can see, the German translation is loaded as expected, while the pop-up dialog has a native look and feel.

# Summary

In this chapter, we compared the advantages and disadvantages of static and dynamic linking. Later on, we used an example application, showing you how to create an installer on Windows and how to package it as a DEB package on Debian Linux. Last but not least, we also learned how to create a redistributable APK file for Android. The slogan, *code less, create more, deploy everywhere* is now fulfilled.

In the next chapter, which is also the last chapter of this book, in addition to how to debug applications, we're also going to look at some common issues and solutions to them.