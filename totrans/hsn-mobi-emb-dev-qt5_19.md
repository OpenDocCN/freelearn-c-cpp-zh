# Building a Linux System

Building your own Linux system for use on an embedded device can be an overwhelming task. Knowing what software is needed to get the stack up and running; knowing what the software dependencies are; finding the software to download and downloading it; configuring, building, and packaging all of that software—it could literally take weeks of time. It used to back in the good old days. Now, there are some great tools to facilitate building a custom Linux filesystem. You can be up and running on an embedded device within a day, if you have a machine powerful enough.

Prototyping is always the first step in device creation. Having the correct tools will streamline this process. Embedded systems need to boot fast and directly into a Qt application, such as an automotive instrument cluster. In this chapter, you will learn about creating a full software stack for embedded Linux systems using Yocto and Boot to Qt for Device Creation. A Raspberry Pi device will be used as a target to demonstrate how to build the operating system.

We will be looking at the following:

*   Bootcamp – Boot to Qt
*   Rolling your own – custom embedded Linux
*   Deploying to an embedded system

# Bootcamp – Boot to Qt

We have already discussed Qt Company's Boot to Qt system in [Chapter 12](143c9219-edf3-4886-aadb-41d91691b2f5.xhtml),* Cross Compiling and Remote Debugging*. Provided with Boot to Qt are configuration files for you to use to create a custom operating system. It requires the BitBake software and the Yocto Project, which is an open source project to help to build custom Linux-based systems, which itself is based on my old friend, OpenEmbedded.

There is a script named `b2qt-init-build-env` in the `/path/to/install/dir/<Qtversion>/Boot2Qt/sources/meta-boot2qt/b2qt-init-build-env` file of this book that will initialize the build for a Raspberry Pi. You run the command from a build directory of your choice.

To get a list of supported devices, use the `list-devices` argument. The output on my system is as follows:

![](img/3a65e42d-5ae7-457d-8cca-6d1899e665b7.png)

You will need to initialize the build system and build environment, so run the script named `b2qt-init-build-env`, which is in the directory that Boot to Qt is installed:

```cpp
/path/to/install/dir/<Qtversion>/Boot2Qt/sources/meta-boot2qt/b2qt-init-build-env init --device raspberrypi3
```

Replace `/path/to/install/dir` with the directory path that `Boot2Qt` is in, typically `~/Qt`. Also, replace `<Qtversion>` to whatever Qt version is installed there. If you are using a different device, change `raspberrypi3` to one that is in the list of supported devices by `Boot2Qt`.

Yocto comes with scripts and configurations so that you can build your own system and customize it, and perhaps add MySQL database support. The `B2Q` script, `setup-environment.sh`, will help to set up the environment for development.

You need to export your device type into the `MACHINE` environmental variable and source the `environment` setup script:

```cpp
export MACHINE=raspberrypi3
source ./setup-environment.sh
```

Now, you can build the default image by using the following command:

```cpp
 bitbake b2qt-embedded-qt5-image
```

You could first customize it by adding a package that you need that isn't there by default—let's say the `mysql` plugin, so that we can access a database remotely! Let's look at how we can do that.

# Rolling your own – custom embedded Linux

 Yocto has a history and got its start from the OpenEmbedded project. The OpenEmbedded project got its name in the programming world from the OpenZaurus project. At that time, I was involved with OpenZaurus and projects surrounding that, with the original focus being the Sharp Zaurus that ran Trolltech's Qtopia using a different operating system. OpenZaurus was an open source replacement OS that users could flash onto their devices. The evolution of the build system went from being the Makefile-based Buildroot to being displaced by BitBake. 

You can, of course, build Poky or Yocto for this section. I am going to use the `Boot2Qt` configurations.

To get started with Yocto so that you can customize it, make a base image by using the following command:

```cpp
bitbake core-image-minimal
```

This will take quite a bit of time.

The basic customization procedure would be the same as customizing Boot to Qt, regarding adding layers and recipes, as well as customizing already existing recipes.

# System customization

By default, the `Boot2Qt rpi` image does not contain the MySQL Qt plugin, so the MySQL example I mentioned previously will not work. I added it by customizing the image build.

Yocto and all BitBake-derived systems use a `conf/local.conf` file so that you can customize the image build. If you do not have one already, after you run `setup-environment.sh file`, create a `local.conf` file and add the following line of code:

```cpp
 PACKAGECONFIG_append_pn-qtbase = " sql-mysql"
```

The `sql-mysql` part comes from Qt's configure arguments, so this is telling `bitbake` to add the `-sql-mysql` argument to the configure arguments, which will build the MySQL plugin and hence include it in the system image. There are other options, but you will need to look in `meta-qt5/recipes-qt/qt5/qtbase_git.bb` and see the lines that start with `PACKAGECONFIG`.

There is one other customization I need to do, which has nothing to do with Qt. OpenEmbedded uses the `www.example.com` URL to test for connectivity. For whatever reason, my ISP's DNS does not have an entry for `https://www.example.com`, so I initially could not reach it, and the build failed straight away. I could have added a new DNS to my computer's network configuration, but it was faster to tell `bitbake` to use another server for its online check, so I added the following line to my `conf/local.conf` file:

```cpp
CONNECTIVITY_CHECK_URIS ?= "https://www.google.com/
```

If you need more extensive customization, you can create your own `bitbake` layer, which is a collection of recipes.

# local.conf file

The `conf/local.conf` file is where you can make local changes to the image build. Like `PACKAGECONFIG_append_pn-`, which we mentioned in the previous section, there are other ways to add packages and issue other configuration commands. The templated `local.conf` has loads of comments to guide you in the process.

`IMAGE_INSTALL_append` allows you to add packages into the image.

`PACKAGECONFIG_append_pn-<package>` allows you to append package-specific configurations to the package. In the case of `qtbase`, it allows you to add arguments to the configure process. Each recipe will have specific configurations.

# meta-<layer> directories

Layers are a way to add packages or add functionality to existing packages. To create your own layer, you will need to create a template directory structure in the `sources/` directory, where you initialized the `bitbake` build. Change `<layer>` to whatever name you are going to use:

```cpp
sources/meta-<layer>/
sources/meta-<layer>/licenses/
sources/meta-<layer>/recipes/
sources/meta-<layer>/conf/layer.conf
sources/meta-<layer>/README
```

The `licenses/` directory is where you put any license files for the package.

Any recipes you may add go into `recipes/` directly. There's more on this a bit later.

The `layer.conf` file is the controlling configuration for the layer. A place to start with this file could be as follows, filled in with generic entries:

```cpp
BBPATH .= ":${LAYERDIR}"
BBFILES += "${LAYERDIR}/recipes-*/*/*.bb \
${LAYERDIR}/recipes-*/*/*.bbappend"
BBFILE_COLLECTIONS += "meta-custom"
BBFILE_PATTERN_meta-custom = "^${LAYERDIR}/"
BBFILE_PRIORITY_meta-custom = "5"
LAYERVERSION_meta-custom = "1"
LICENSE_PATH += "${LAYERDIR}/licenses"
```

Change `meta-custom` to whatever you want to name it.

Once you have created the layer, you will need to add it to the `conf/bblayers.conf` file, which is in the directory that you initialized in the `Boot2Qt` build. In my case, this was `~/development/b2qt/build-raspberrypi3/`.

We can now add one or more packages to our custom layer.

# <recipe>.bb files

You can also create your own recipe if you have existing code, or if there is a software project somewhere that you want to include in the system image.

In the custom layer, we created a `recipes/` directory where our new recipe can live.

To get a feel of how recipes can be written, take a look at some of the recipes included with Boot2Qt or Yocto.

There are some scripts that can help in the creation of recipe files, that is, `devtool` and `recipetool`. The `devtool` and `recipetool` commands are fairly similar in what they do. `Devtool` makes it easier if you need to apply patches and work on the code. Sometimes, your software needs to be developed or debugged on the actual device, for example, if you are developing something that uses any sensors. `Devtool` can also build the recipe so that you can work the kinks out.

# devtool command

The output of `devtool --help` is as follows:

![](img/5ae324eb-3713-4e94-ab61-f35985343cc9.png)

The most important arguments would be `add`, `modify`, and `upgrade`.

For `devtool`, I will use a Git repository URL to add my repository of `sensors-examples`:

```cpp
devtool add sensors-examples https://github.com/lpotter/sensors-examples.git
```

Running the preceding command will output something similar to this:

![](img/fb60f5a5-b026-4e64-967e-45defe194f94.png)

 We need to try and build the package to see whether it succeeds or fails, which we can do by running the following command:

```cpp
devtool build sensors-examples

```

We might need to edit this `.bb` file to make it build if it does fail.

In the case of `sensors-examples`, we will get the following output:

![](img/3a278aac-8a3e-4cc5-bfd6-dc40f6205b00.png)

We've built it!

You will find this build in `tmp/work/cortexa7hf-neon-vfpv4-poky-linux-gnueabi/sensors-examples/1.0+git999-r0`.

If you want to edit a recipe, you can use `devtool` and then create a patch so that you can use it:

```cpp
devtool modify qtsensors
```

We get the following output by running the preceding command:

![](img/f432ca34-25e5-4212-9a88-4743b923bba0.png)

This will duplicate the recipe in your local workspace so that you can edit without losing it when you update `bitbake`. Now, you can edit the sources, in this case, of `qtsensors`. I have a patch to add a `qtsensors` plugin for the Raspberry Pi Sense HAT, so I am going to manually apply that now:

![](img/629745f4-a2dd-43e1-882c-d483d6ebb9b9.png)

My patch is old, and I need to fix it up. You can build this on its own by running the following command:

```cpp
devtool build qtsensors
```

This initially fails to find `rtimulib.h`, so we need to add a dependency on that lib.

In the OpenEmbedded Layer Index, there is a `python-rtimu` recipe, but it does not export the headers or build the library, so I will create a new recipe based on the Git repository, as follows:

```cpp
devtool add rtimulib https://github.com/RPi-Distro/RTIMULib.git
```

This is a `cmake`-based project, and I will need to modify the recipe to add some `cmake` arguments. To edit this, I can simply run the following command:

```cpp
devtool edit-recipe rtimulib

```

I added the following lines, which use `EXTRA_OECMAKE` to disable some of the demos that depend on Qt 4\. I think at one time, I had a patch that ported it to Qt 5, but I cannot find it. The last `EXTRA_OEMAKE` tells `cmake` to build in the Linux directory. Then, we tell `bitbake` it needs to inherit the `cmake` stuff:

```cpp
EXTRA_OECMAKE = "-DBUILD_GL=OFF"
EXTRA_OECMAKE += "-DBUILD_DRIVE=OFF"
EXTRA_OECMAKE += "-DBUILD_DRIVE10=OFF"
EXTRA_OECMAKE += "-DBUILD_DEMO=OFF"
EXTRA_OECMAKE += "-DBUILD_DEMOGL=OFF"
EXTRA_OECMAKE += "Linux"
inherit  cmake
```

We then need to edit our `qtsensors_git.bb` file so that we can add a dependency on this new package. This will allow it to find the headers:

```cpp
DEPENDS += "rtimulib"

```

When I run the build command, `bitbake qtsensors`, it will make sure my `rtimulib` package is built, then apply my `sensehat` patch to `qtsensors`, and then build and package that up!

`Recipetool` is another way that you can create a new recipe. It is simpler in design and usage than `devtool`.

# recipetool command

The output of using the `recipetool create --help` command is as follows:

![](img/0a9d2928-4ba6-48cb-8176-499524e2fc1b.png)

As an example, I ran `recipetool -d create -o rotationtray_1.bb https://github.com/lpotter/rotationtray.git`:

![](img/3ac573f3-1f38-466f-9019-d6068005e820.png)

Using the `-d argument` means it will be more verbose, so I excluded some output:

![](img/ff47533d-54dd-4ff3-bf0f-368efae25983.png)

Now, you may want to edit the resulting file. A great way to learn about `bitbake` is to look at other recipes and see how they do things.

# bitbake-layers

OpenEmbedded comes with a script called `bitbake-layers`, which you can use to get information on layers that are available. You can also use this to add a new layer or remove one from the configuration file.

Running `bblayers --help` will give us the following output:

![](img/340262f5-fe17-4b58-82ac-cde9695e40ad.png)

Running `bitbake-layers show-recipes` will dump all of the available recipes. This list can be quite extensive. 

# yocto-layer

Yocto has a script named `yocto-layer`, which will create an empty layer directory structure that you can then add with `bitbake-layers`. You can also add an example recipe and a `bbappend` file.

To create a new layer, run `yocto-layer` with the `create` argument:

```cpp
yocto-layer create mylayer
```

This will run interactively and ask you a few questions. I told it yes to both questions to create examples:

![](img/57244712-3fc3-43bd-bac4-d34ef7d837a9.png)

You will then see a new directory tree named `meta-mylayer`. You can then make the new layer available to `bitbake` using `bitbake-layers`, as follows:

```cpp
bitbake-layers add-layer meta-mylayer
```

Use the following command to see the new layer running:

```cpp
bitbake-layers show-layers
```

# bbappend files

When I imported the `qtsensors` recipe into my workspace, I could have used a `bbappend` file. When you import the recipe into your workspace, it is essentially being duplicated. Please note, however, that you will no longer be able to build it with `devtool`.

I also mentioned that the `yocto-layer` script can create an example `bbappend` file with a patch so that we can see how it works. The filename that you choose must match whatever recipe you are modifying. The only difference in name would be the extension, which is `.bbappend` for a `bbappend` file.

In the `workspace/conf/local.conf` file, there is a line about BBFILES that tells me where it is looking for `bbappend` files. Of course, you can put them anywhere as long as you tell `bitbake` where they are. Mine is configured for them to be in `${LAYERDIR}/appends/*.bbappend`.

Ours is simple—it only applies a patch. With only the following few lines in the `bbappend` file, we can get it up and running:

*   `SUMMARY`: Simple string explaining the patch
*   `FILESEXTRAPATHS_prepend`: String of the path to the patch files
*   `SRC_URI`: The URL string to the patch file

If we wanted to create a `bbappend` file to patch `qtsensors` with the `sensehat` patch, it would be a four line edit, along with the actual patch. The simple `bbappends` file would look like this:

```cpp
SUMMARY = "Sensehat plugin for qtsensors"
DEPENDS += "rtimulib"
FILESEXTRAPATHS_prepend := "${THISDIR}:"
SRC_URI += "file://0001-Add-sensehat-plugin.patch"
```

It's good practice to place the patch into a directory, but this one is on the same level as the `bbappend` file.

We would need to remove the `qtsensors` recipe that was imported from the workspace before we can build our appended recipe:

```cpp
devtool reset qtsensors
```

Place `qtsensors_git.bbappend` and the patch file into the `appends` directory. To build it, simply run the following command:

```cpp
bitbake qtsensors
```

Now that we can customize our OpenEmbedded/Yocto image, we can deploy to the device.

# Deploying to an embedded system

We've built a custom system image, and there are a few different ways that systems can be deployed onto a device. Usually, an embedded device has a particular way of doing this. The image can be deployed to a Raspberry Pi by writing the system image file directly onto a storage disk using dd or similar. Other devices might need to be deployed by writing the filesystem to a formatted disk, or even as low level as using JTAG.

# OpenEmbedded

If you plan on using Qt with OpenEmbedded, you should be aware of the ​`meta-qt5-extra` layer, which contains desktop environments such as LXQt and even KDE5\. I personally use both environments and switch back and forth between the two on my desktop, but I prefer LXQt most of the time as it's lightweight.

Building an OpenEmbedded image with LXQt is fairly straightforward, and similar to building a Boot to Qt image.

To see the image targets that are available, you can run the following command:

```cpp
bitbake-layers show-recipes | grep image

```

If you have Boot to Qt, you should see the `b2qt-embedded-qt5-image` layer, which we will use to create the image for Raspberry Pi. You should also see OpenEmbedded's `core-image-base and core-image-x11`, which may also be interesting.

There are other layers available that you can search for and download from [https://layers.openembedded.org/layerindex/branch/master/layers/](https://layers.openembedded.org/layerindex/branch/master/layers/).

The deployment method really depends on your target device. Let's see how we can deploy the system image to a Raspberry Pi.

# Raspberry Pi

The example in this section targets the Raspberry Pi. You may have a different device, and the process here might be similar.

If you intend to only create a cross `toolchain` that you can use in Qt Creator, you can run the following command:

```cpp
bitbake meta-toolchain-b2qt-embedded-qt5-sdk
```

To create the system image to copy to an SD card, run the following command:

```cpp
bitbake b2qt-embedded-qt5-image
```

The `b2qt-embedded-qt5-image` target will also create the SDK if it is needed. When you let that run for a day or so, you'll have a freshly baked Qt image! I would suggest using the fastest machine you have with the most memory and storage, as a full distro build can take many hours, even on a fast machine.

You can then take the system image and use the device's flash procedure or whatever method it uses to make the filesystem. For the RPI, you put the micro SD into a USB reader, and then run the `dd` command to write the image file.

The resulting file I need to write to the SD card was at the following location:

```cpp
tmp/deploy/images/raspberrypi3/b2qt-embedded-qt5-image-raspberrypi3-20190224202855.rootfs.rpi-sdimg
```

To write this to the SD card, use the following command:

```cpp
sudo dd if=/path/to/sdimg of=/dev/path/to/usb/drive bs=4M status=progress
```

My exact command was as follows:

```cpp
sudo dd if=tmp/deploy/images/raspberrypi3/b2qt-embedded-qt5-image-raspberrypi3-20190224202855.rootfs.rpi-sdimg of=/dev/sde bs=4M status=progress
```

Now, wait until everything has been written to the disk. Plop it into the Raspberry Pi SD slot, power it on, and then you're on your way!

# Summary

In this chapter, we learned about how to use `bitbake` to build a custom system image, starting with Qt's Boot to Qt configuration files. The process is similar to building Yocto, Poky, or Ångström. We also learned how use `devtool` to customize Qt's build to add more functionality. Then, we discussed how to add your own recipe using `recipetool` into the image. By doing this, you were also able to add this recipe into a new layer. We finished off by deploying the image onto an SD card so that it could be run on the Raspberry Pi.