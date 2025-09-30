# Universal Platform for Mobiles and Embedded Devices

Deploying applications and targeting all the different platforms can take heaps of time and cost thousands of dollars. There's a new target platform for Qt applications called Qt for WebAssembly that allows Qt and Qt Quick apps to be run over a network from a web browser. You will learn how to set up, cross build, deploy, and run Qt applications that work on any device with a modern web browser. You could say that Qt for WebAssembly is the universal platform.

We will detail the following material:

*   Technical requirements
*   Getting started
*   Building with the command line
*   Building with Qt Creator
*   Deploying for mobile and embedded devices
*   Tips, tricks and suggestions

# What is this WebAssembly thing?

WebAssembly is neither strictly Web nor Assembly. At the same time, it is a little of both.

At the technical level, it is a new binary instruction format for a stack-based virtual machine, according to the WebAssembly web site at [http://webassembly.org](http://webassembly.org). It runs in a modern web browser, but people are naturally experimenting with this and it can now run standalone and experimentally like any other app, with support being written for the Linux kernel.

Through the use of the Emscripten tool, it can be compiled from C and C++. Emscripten is a tool written in Python that uses LLVM to transpile C++ code into WebAssembly byte code that can be loaded by a web browser.

WebAssembly byte code runs in the same sandbox as JavaScript, so consequentially it has the same limitations regarding access to the local file system, as well as living in one thread. It also has the same security benefits. Although there is work being done to fully support pthreads, it is, at the time of this writing, still experimental.

# Technical requirements

Easy install binary SDK from the following Git repository:

*   Emscripten sdk [https://github.com/emscripten-core/emscripten.git](https://github.com/emscripten-core/emscripten.git)

Alternatively, manually compile the SDK. You can download the sources from these Git URLs:

*   Emscripten [https://github.com/emscripten-core/emscripten.git](https://github.com/emscripten-core/emscripten.git)
*   Binaryen [https://github.com/WebAssembly/binaryen.git](https://github.com/WebAssembly/binaryen.git)
*   LLVM [https://github.com/llvm/llvm-project.git](https://github.com/llvm/llvm-project.git)

# Getting started

According to the Emscripten website at [https://emscripten.org/:](https://emscripten.org/)

*Emscripten is a toolchain that uses LLVM to transpile code to WebAssembly to run in a web browser at near native speeds.*

These are the two ways to install Emscripten: 

*   Clone the repository, install precompiled binaries
*   Clone the repositories, build them

I recommend the first one, as LLVM is very time-consuming to build. It is also recommended to use Linux or macOS. If you are on Windows, you can install the Linux subsystem and use that, or use MinGW compiler. The Visual Studio compiler does not seem to support output targets with the four-letter extensions that Emscripten outputs, namely `.wasm` and `.html`.

# Download Emscripten

You need to have Git and Python installed for this—just clone the `emscripten sdk`:

```cpp
git clone https://github.com/emscripten-core/emscripten.git.
```

In there, are Python scripts to help out, the most important one being `emsdk`.

First run `./emsdk --help` to print out some documentation on how to run it.

Then you need to install and then activate the SDK as follows:

```cpp
./emsdk install latest
./emsdk activate latest
```

You can target a specific SDK; you can see what is available by running the following command:

```cpp
./emsdk list
```

Then install a particular version of the SDK by running the following commands:

```cpp
./emsdk install sdk-1.38.16-64bit
./emsdk activate sdk-1.38.16-64bit
```

The `activate` command sets up the `~/.emscripten` file that contains the environment settings needed by Emscripten.

To be able build with it, you need to source the `emsdk_env.sh` file as follows:

```cpp
source ~/emsdk/emsdk_env.sh
```

Qt targets a certain Emscripten version that is known to be good for that version. For Qt 5.11, Qt for WebAssembly has its own branch—`wip/webassembly`. It has been integrated into 5.12 as a tech preview, and in 5.13 for official support. At the time of this writing, it is planned to be included with Qt Creator as a binary install.

# Building an Emscripten SDK manually

If you want to build Emscripten manually, such as to compile upstream LLVM which has support for transpiling directly to WebAssembly binary instead of writing first to JavaScript and then to WebAssembly. This can speed up compile times, but, at the time of this writing, is still experimental. This makes use of adding an argument to the linker `-s WASM_OBJECT_FILES=1`. 

For more information on using `WASM_OBJECT_FILES`, see [https://github.com/emscripten-core/emscripten/issues/6830](https://github.com/emscripten-core/emscripten/issues/6830).

# Technical requirements

You will need to install `node.js` and `cmake` packages from your OS. Clone the following resources:

```cpp
mkdir emsdks
cd emsdks
git clone -b 1.38.27 https://github.com/kripken/emscripten.git
git clone -b 1.38.27 https://github.com/WebAssembly/binaryen.git
git clone https://github.com/llvm/llvm-project.git
```

Emscripten does not have to be built, as it is written in Python.

To build `binaryen`, enter the following code:

```cpp
cd binaryen
cmake .
make
```

To build LLVM, enter the following code:

```cpp
mkdir llvm
cmake ../llvm-project/llvm -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi;lld" -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=WebAssembly -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=WebAssembly

make
```

Run `emscripten` to write the configure file as follows:

```cpp
cd emscripten
./emcc --help
```

This will create a `~/.emscripten` file. Copy this file over to your `emsdks` directory as follows:

```cpp
cp ~/.emscripten /path/to/emsdks/.emscripten-vanillallvm
```

To set up the environment, write a script as follows:

```cpp
#!/bin/bash
SET EMSDK=/path/to/emscripten
SET LLVM=/path/to/llvm/bin
SET BINARYEN=/path/to/binaryen
SET PATH=%EMSDK%;%PATH%
SET EM_CONFIG=/path/to/emsdks/.emscripten-vanillallvm
SET EM_CACHE=/path/to/esdks/.emscripten-vanillallvm_cache
```

Save it somewhere as `emsdk-env.sh`.

You will need to make this executable, so run `chmod +x emsdk-env.sh`.

Whenever you need to set up the build environment, simply run this script and use the same console to build.

Now that we are ready, let's see how to configure and build Qt.

# Configuring and compiling Qt

You can find information on Qt for WebAssembly at this URL: [https://wiki.qt.io/Qt_for_WebAssembly](https://wiki.qt.io/Qt_for_WebAssembly)

I guess we need the sources. You can get them through Qt Creator, or you can `git clone` the repository. Using Git, you have more control over which version and any branch if needed.

For 5.12 and 5.13, you can simply clone the following tag:

```cpp
git clone http://code.qt.io/qt/qtbase.git -b v5.12.1
```

Alternatively, you can `clone` this tag:

```cpp
git clone http://code.qt.io/qt/qtbase.git -b v5.13.0
```

As with any new technology, it is moving fast, so grab the latest version you can. For this book, we are using Qt 5.12, but I included mentioning other versions as they have many bug fixes and optimizations.

Now we can configure and compile Qt!

For 5.12 and 5.13 it was simplified to the following:

```cpp
configure -xplatform wasm-emscripten -nomake examples -nomake tests
```

If you need threads, 5.13 has support for multithreading WebAssembly, but you also need to configure the browser to support it.

Once it configures, all you need to do is run make!

Then, to build your Qt app for running in a web browser, simply use the `qmake` command from the build directory and run it on your apps pro file. Not every Qt feature is supported—like local filesystem access and threads. `QOpenGLWidget` is also not supported, although `QOpenGLWindow` works fine. Let's see how to build using then command line.

# Building with the command line

Building a Qt for a WebAssembly application requires you to source the Emscripten environment file, so run this in your console command as follows:

```cpp
source /path/to/emsdk/emsdk_env.sh
```

You will need to add the path to Qt for WebAssembly `qmake` as follows:

```cpp
export PATH=/path/to/QtForWebAssembly/bin:$PATH.
```

Of course, you must replace `/path/to` with the actual filesystem path.

You are then ready for action! You build it just like any other Qt app, by running `qmake` as follows:

```cpp
qmake mycoolapp.pro && make.
```

If you need to debug, rerun `qmake` with `CONFIG+=debug` as follows:

```cpp
qmake CONFIG+=debug mycoolapp.pro && make.
```

This will add various Emscripten specific arguments to the compiler and linker.

Once it is built, you can run it by using the `emrun` command from Emscripten, which will start a simple web server and serve the `<target>.html` file. This will, in turn, load up `qtloader.js`, which, in turn, loads up the `<target>.js` file, which loads the `<target>.wasm` binary file:

```cpp
emrun --browser firefox --hostname 10.0.0.4 <target>.html.
```

You can also give `emrun` the directory, such as:

```cpp
emrun --browser firefox --hostname 10.0.0.4 ..
```

This gives you time to bring up the browser's web console for debugging. Now, let's see how to use Qt creator for building.

# Building with Qt Creator

It is possible to use Qt Creator to build and run your Qt app once you have compiled Qt itself from the command line.

# The Build environment

In Qt Creator, navigate to Tools | Options... | Kits

Then go to the Compilers tab. You need to add `emcc` as a C compiler, and `em++` as a C++ compiler, so click on the Add button and select Custom from the drop-down list.

First select C and add the following details: 

*   Name: `emcc (1.38.16)`
*   Compiler path: `/home/user/emsdk/emscripten/1.38.16/emcc`
*   Make path: `/usr/bin/make`
*   ABI: `x86 linux unknown elf 64bit`
*   Qt mkspecs: `wasm-emscripten`

Select C++ and add the following details:

*   Name: `emc++(1.38.16)`
*   Compiler path: `/home/user/emsdk/emscripten/1.38.16/em++`
*   Make path: `/usr/bin/make`
*   ABI: `x86 linux unknown elf 64bit`
*   Qt mkspecs: `wasm-emscripten`

Click Apply.

Go to the tab labeled Qt Versions and click on the Add button. Navigate to where you build Qt for WebAssembly, and, in the `bin` directory, select the qmake. Click Apply.

Go to the tab labeled Kits, and click on the Add button. Add the following details:

*   Name: `Qt %{Qt:Version} (qt5-wasm)`
*   Compiler: `C: emcc (1.38.16`
*   Compile: `C++: em++ (1.38.16)`
*   Qt version: `Qt (qt5-wasm)`

# The Run environment

You need to make your application's project active to build for Qt for WebAssembly. From the left-hand side buttons in Qt Creator, select Projects, and select your Qt for WebAssembly kit.

Running the WebAssembly apps in Qt Creator is currently a bit tricky, as you need to specify `emrun` as a custom executable and then the build directory or `<target>.html` file as its argument. You can also specify which browser to run. You can run Chrome using the `--browser chrome argument`.

To get a list of found browsers, run the command `emrun --list_browsers`.

You can even run the app in an Android device that is connected to a USB using the `--android` argument. You need to have **Android Debug Bridge** (**adb**) command installed and running.

Anyway, now that we know how to run the app, we need to tell the Qt Creator project to run it. 

Go to Projects | Run. In the Run section, select Add | Custom Executable and add the following details:

*   Executable: `/home/user/emsdk/emrun <target>.html`
*   Working directory: `%{buildDir}`

Now we are ready to build and run. Here is how it should look:

![](img/0993db3c-14f7-43b0-b7d7-988382dca979.png)

We can even run OpenGL apps! Here is the `hellogles3` example from Qt running in the Firefox browser on Android:

![](img/787ad667-750d-4c79-979b-a00035465efb.png)

We can also run declarative apps! Here is Qt Quick's `qopenglunderqml` example app:

![](img/11a3c9d8-4ab8-4914-b5b6-c7010ce5e66f.png)

# Deploying for mobile and embedded devices

Really, deploying for mobile and embedded devices is only copying the resulting files from Emscripten built onto a CORS-enabled web server.

Any web browser that supports WebAssembly will be able to run it. 

Of course, there are considerations regarding screen size.

For testing, you can run your application using the `emrun` command from the Emscripten SDK. If you plan on testing from another device other than localhost, you will need to use the `--hostname` argument to set the IP address that it uses.

There are Python scripts for CORS-enabled web servers for testing as well. The Apache web server can also be configured to support CORS. 

There are five files that currently need to be deployed—`qtloader.js`, `qtlogo.svg`, `<target>.html`, `<target>.js`, and `<target>.wasm`. The `.wasm` file is the big WebAssembly binary, statically linked. Following are few suggestions to help you along with the process.

# Tips, tricks, and suggestions

Qt for WebAssembly is treated by Qt as a cross platform build. It is an emerging technology and, as such, some features required may need special settings configuration to be changed or enabled. There are a few things you need to keep in mind when using it as a target.

Here, I run through some tips regarding Qt for WebAssembly.

# Browsers

All major browsers now have support for loading WebAssembly. Firefox seems to load fastest, although Chrome has a configuration that can be set to speed it up (look at `chrome://flags for #enable-webassembly-baseline`). Mobile browsers that come with Android and iOS also work, although these may run into out of memory errors, depending on the application being run.

Qt 5.13 for WebAssembly has added experimental support for threads, which rely `onSharedArrayBuffer` support in the browsers. This has been turned off by default, due to Spectre vulnerabilities, and need to be enabled in the browsers.

In Chrome, navigate to `chrome://flags` and enable `#enable-webassembly-threads`.

In Firefox, navigate to `about://config` and enable `javascript.options.shared.memory`.

# Debugging

Debugging is done by using the debugging console in the web browser. Extra debugging capabilities can be enabled by invoking `qmake` with `CONFIG+=debug`, even with a Qt compiled in release mode. Here is what a crash can look like: 

![](img/e4f39e1f-2c55-4d87-89f0-8f969d470f2b.png)

You can also remote debug from your phone and see the remote browser's JavaScript console output on your desktop. See the following link:

 [https://developer.mozilla.org/en-US/docs/Tools/Remote_Debugging](https://developer.mozilla.org/en-US/docs/Tools/Remote_Debugging)

# Networking

Simple download requests can be made with the usual `QNetworkAccessManager`. These will go through `XMLNetworkRequest`, and will require a CORS-enabled server to download from. Typical `QTCPSocket` and `QUdpSockets` get transpiled into WebSockets. Your web server needs to support WebSockets, or you can use the Websockify tool, which is available from the following link:

[https://github.com/novnc/websockify](https://github.com/novnc/websockify)

# Fonts and filesystem access

System fonts cannot be accessed, and must be included and embedded into the application. Qt embeds one font.

Filesystem access is also not currently supported, but will be in the future by using a Qt WebAssembly specific API.

# OpenGL

OpenGL is supported as OpenGL ES2, which gets transpiled into WebGL.

There are a few differences between OpenGL ES2 and WebGL that you should be aware of if you plan on using OpenGL in your WebAssembly application. WebGL is more strict generally. 

Here are some of the differences for WebGL:

*   A buffer may only be bound to one `ARRAY_BUFFER` or `ELEMENT_ARRAY_BUFFER` in it's lifetime
*   No client side `Arrays`
*   No binary shaders, `ShaderBinary`
*   Enforces `offset` for `drawElements`; `vertexAttribPointer` and `stride` arguments for `vertexAttribPointer` are a multiple of the size of the data type

*   `drawArrays` and `drawElements` are restricted from requesting data outside the bounds of a buffer
*   Adds `DEPTH_STENCIL_ATTACHMENT` and `DEPTH_STENCIL`
*   `texImage2D` and `texSubImage2D` size based on the `TexImageSource` object
*   `copyTexImage2D`, `copyTexSubImage2D` and `readPixels` cannot touch pixels outside of `framebuffer`
*   Stencil testing and bound `framebuffer` have restricted drawing
*   `vertexAttribPointer` value must not exceed the value 255
*   `zNear` cannot be greater than `zFar`
*   constant color and constant alpha cannot be used with `blendFunc`
*   no support for `GL_FIXED`
*   `compressedTexImage2D` and `compressedTexSubImage2D` are not supported
*   GLSL token size limited to 256 characters
*   GLSL is ASCII characters only
*   GLSL limited to 1 level of nested structures
*   Uniform and attribute location lengths limited to 256 characters
*   `INFO_LOG_LENGTH`, `SHADER_SOURCE_LENGTH`, `ACTIVE_UNIFORM_MAX_LENGTH`, and `ACTIVE_ATTRIBUTE_MAX_LENGTH` have been removed.
*   Texture type passed to `texSubImage2D` must match `texImage2D`
*   Calls that `read` and `write` to same texture (feedback loop) not allowed
*   Reading data from missing attachment is not allowed
*   Attribute aliasing not allowed
*   ​`gl_Position` initial value defined as (0,0,0,0)

For more information, see the following web pages:

WebGL 1.0 [https://www.khronos.org/registry/webgl/specs/latest/1.0/#6](https://www.khronos.org/registry/webgl/specs/latest/1.0/#6)

WebGL 2.0 [https://www.khronos.org/registry/webgl/specs/latest/2.0/#5](https://www.khronos.org/registry/webgl/specs/latest/2.0/#5)

# Supported Qt modules

Qt for WebAssembly supports the following Qt modules:

*   `qtbase`
*   `qtdeclarative`
*   `qtquickcontrols2`
*   `qtwebsockets`
*   `qtmqtt`
*   `qtsvg`
*   `qtcharts`
*   `qtgraphicaleffects`

# Other caveats

Secondary event loops do not work in Qt for Webassembly. This is because the Emscripten event loop it needs to tie in to does not return. If you need to pop up a dialog, do not call `exec()` but call `show()`, and use signals to get a return value.

Virtual keyboards on mobile platforms like Android and iOS do not automatically pop up. You can use Qt Virtual Keyboard directly in your project.

# Summary

Qt for WebAssembly is a new and upcoming platform for Qt, which runs Qt apps in a web browser.

You should now be able to download or build the Emscripten SDK, and use to build Qt for WebAssembly. You can now run Qt apps in a web browser, including mobile and embedded devices, as long as the browser supports WebAssembly.

In the final chapter, we explore building a complete Linux embedded operating system.