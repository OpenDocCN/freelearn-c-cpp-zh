# 第八章：使用超级构建简化入门

在本章中，我们将回到简化和精简项目设置的工作中。在开发过程中，添加功能和应对随之而来的复杂性之间总有一种自然的推拉关系。在*第七章*《为你的库添加安装支持》中，我们花了大量时间切换目录并运行 CMake 命令。为了构建我们的应用程序，我们需要穿越至少五个文件夹（`third-party`、`array`、`draw`、`gol` 和 `app`），并在途中运行大量 CMake 命令。这是学习 CMake 的一种极好的方式，但当你想要完成工作时，这并不有趣。它还可能阻碍不熟悉的用户访问或贡献你的项目。

现在是时候解决这个问题了。你将在本章中学到的技能将有助于减少启动和运行项目所需的手动步骤。本章将向你展示如何去除平台特定的脚本，并自动化更多的构建过程。

在本章中，我们将介绍以下主要内容：

+   使用`ExternalProject_Add`与你自己的库

+   配置超级构建

+   使用 CMake 自动化脚本

+   在嵌套文件中设置选项

+   安装应用程序

# 技术要求

要跟随本教程，请确保已满足*第一章*《入门》的要求。这些要求包括以下内容：

+   一台运行最新**操作**系统（**OS**）的 Windows、Mac 或 Linux 机器

+   一个可用的 C/C++ 编译器（如果你还没有，建议使用每个平台的系统默认编译器）

本章中的代码示例可以通过以下链接找到：[`github.com/PacktPublishing/Minimal-CMake`](https://github.com/PacktPublishing/Minimal-CMake)。

# 使用 ExternalProject_Add 与你自己的库

在上一章中，我们主要依靠手动 CMake 构建和安装命令（以及一些 CMake 预设的帮助）来增加我们对 CMake 的熟悉度，并以稍低的抽象层次工作，以理解 CMake 在后台所做的事情。现在我们对这些概念已经更加熟悉，是时候去除在每个单独的库文件夹中导航并运行以下熟悉的 CMake 命令的乏味工作了：

```cpp
cmake --preset <preset-name>
cmake --build <build-folder> --target install
```

我们可以开始更新项目，利用 CMake 提供的更多有用功能。首先，我们将更新现有的第三方 `CMakeLists.txt` 文件，不仅引入 SDL 2 和 bgfx，还包括我们创建并依赖的库。这将消除我们手动安装这些库的需要，并允许我们运行一对 CMake 命令（配置和构建/安装）来获取我们所需的所有依赖项，以支持我们的*生命游戏*应用程序。

让我们首先查看`ch8/part-1/third-party/CMakeLists.txt`。该文件与之前大致相同，只是在现有的`ExternalProject_Add`命令下，我们添加了对位于`ch8/part-1/lib`中的库的引用。

这是`mc-array`库的示例：

```cpp
ExternalProject_Add(
  mc-array
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib/array
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/mc-array-build/${build_type_dir}
  INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install
  CMAKE_ARGS ${build_type_arg} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  CMAKE_CACHE_ARGS -DCMAKE_DEBUG_POSTFIX:STRING=d)
```

该命令看起来与我们在*第六章*中讲解的非常相似，*安装依赖项和 ExternalProject_Add*。唯一的真正不同之处是对`SOURCE_DIR`的引用。由于本书仓库的布局稍显不传统，我们可以直接引用源文件夹，因为库的源代码存储在同一仓库中：

```cpp
SOURCE_DIR URL or GIT_REPOSITORY link. If we did, for some reason, want to refer to an older version of one of the libraries at a specific moment in the Git history of our project, we could use this approach:

```

ExternalProject_Add(

mc-array

GIT_REPOSITORY https://github.com/PacktPublishing/Minimal-CMake.git

GIT_TAG 18535c9d140e828895c57dbb39b97a3307f846ab

SOURCE_SUBDIR ch8/part-1/lib/array

...

```cpp

			We did the same thing in *Chapter 3*, *Using FetchContent with External Dependencies*, when using `FetchContent`. The preceding command will clone the entire repository into the build folder of our third-party `CMakeList.txt` file, and then treat the `ch8/part-1/lib/array` directory as the root of the repository (at least as far as `ExternalProject_Add` is concerned). It’s not often needed but can be useful if a repository holds more than one CMake project.
			While we’re making changes to `third-party/CMakeLists.txt`, we’ll also make one small improvement to how we handle our bgfx dependency. When bgfx was first introduced in *Chapter 6*, *Installing Dependencies and ExternalProject_Add* (see `ch6/part-4`), we wound up needing to clone the repository twice, once for the static version of the library (needed to build the tools), and again for the shared version of the library that our application linked against. The good news is there’s a handy technique we can apply to download the library once. The following is an extract of the changes with the differences highlighted:

```

ExternalProject_Add(

bgfxt

GIT_REPOSITORY https://github.com/bkaradzic/bgfx.cmake.git

GIT_TAG v1.127.8710-464

...

ExternalProject_Get_Property(bgfxt SOURCE_DIR)

ExternalProject_Add(

bgfx

URL "file://${SOURCE_DIR}"

DEPENDS bgfxt

...

```cpp

			The first `ExternalProject_Add` call is the same as before; we let it know the repository and specific `GIT_TAG` to download. However, in the second version, instead of repeating those lines, we first call `ExternalProject_Get_Property`, passing the `bgfxt` target and the `SOURCE_DIR` property. `SOURCE_DIR` will be populated with the location of the `bgfxt` source code once it’s downloaded, and in the second command, we point the `bgfx` target to reference that source code location. As the code is identical and it’s just how we’re building it that’s different, this saves a bit of time and network bandwidth downloading the same files all over again.
			You might have noticed that we’re using `URL "file://${SOURCE_DIR}"` as opposed to `SOURCE_DIR ${SOURCE_DIR}`. This is because if we tried to use `SOURCE_DIR`, the `ExternalProject_Add` command would fail at configure time because `SOURCE_DIR` is expected to already be present when we configure. As this isn’t the case with our dependency (the source for `bgfxt` will only be downloaded and made available at build time), we can use the `URL` option with a local file path shown by `file://`, which will cause the file path to instead be resolved at build time.
			With the addition of the three new `ExternalProject_Add` calls referencing our libraries, when building our project, we only need to visit two directories, as opposed to the earlier five. We can navigate to `ch8/part-1/third-party` and run the following CMake commands:

```

cmake -B build -G "Ninja Multi-Config"

cmake --build build --config Release

```cpp

			This will download, build, and install all our dependencies at once. We then only need to navigate to `ch8/part-1/app` and run the following:

```

cmake --preset multi-ninja

cmake --build build/multi-ninja --config Release

```cpp

			This will build and link our application. We’ve also tidied up our `CMakePresets.json` file to have `CMAKE_PREFIX_PATH` only refer to `${sourceDir}/../third-party/install`, as opposed to the numerous install folders we had for each of our internal dependencies in the last chapter.
			Finally, to launch the application, we first need to compile our shaders and then launch the application from our application root directory (`ch8/part-1/app`):

```

./compile-shader-<platform>.sh/bat

./build/multi-ninja/Release/minimal-cmake_game-of-life_window

```cpp

			This is a substantial improvement from before, but we can do better. The main build is still split across two stages (dependencies and application), and we still have the annoying issue of needing to remember to compile the shaders (an easy step to overlook). We want to achieve the holy grail of one command to bootstrap everything. Let’s start by seeing how we can solve the first problem by reducing our build steps from two to one.
			Configuring a super build
			For those unfamiliar with the term **super build**, it is a pattern to bundle external dependencies and the local build together in one step. Behind the scenes, things are still being built separately, but from the user’s perspective, everything happens at once.
			Super builds are great for getting up and running with a project quickly. They essentially automate all the configuration and dependency management for you. One other useful quality about them is they’re opt-in. If you want to build the dependencies yourself and install them in a custom location, letting the application explicitly know where to find them, that’s still possible, and the super build won’t get in the way.
			Super builds provide the best of both worlds. They are a singular way to simply build a project, and they can also be easily disabled, allowing you to configure your dependencies as you see fit.
			Integrating super build support
			We’re going to walk through the changes necessary to add super build support by reviewing `ch8/part-2`. All changes are confined to the `app` subfolder.
			To begin with, to have our app feel a bit more like a real CMake project (where our CMake project is the root folder), we’re going to move the `third-party` folder inside `app`. The structure now looks like this:

```

.

├── app

│   └── third-party

└── lib

```cpp

			This is compared to how it was before:

```

.

├── app

├── third-party

└── lib

```cpp

			This is more representative of a real CMake project and will make enabling and disabling super builds a bit easier.
			We’ll start by looking at the changes in `ch8/part-2/app/CMakeLists.txt`. The first and only changes are right at the top of the file:

```

option(SUPERBUILD "执行超级构建（或不执行）" OFF)

if(SUPERBUILD)

add_subdirectory(third-party)

return()

endif()

```cpp

			The first change is simply the addition of a new CMake option to enable or disable super builds. It’s defaulted to `OFF` for now, but it’s easy to change, and we, of course, have some new CMake presets we’ll cover later in this section, which provide a few different permutations people might want to use.
			Next comes the new functionality that only runs if super builds are enabled. It’s worth emphasizing that everything in our application’s `CMakeLists.txt` file stays exactly the same as before. We can easily revert to the earlier way of building if we wish to by simply setting `SUPERBUILD` to `OFF`. Even using both at the same time is easy and convenient (we’ll get into how to do this a bit later).
			Inside the super build condition, we call `add_subdirectory` on the `third-party` folder (this was one of the reasons we moved it inside the `app` folder to make composing our `CMakeLists.txt` scripts a little easier). In this instance, we could have used `include` instead of `add_subdirectory`. However, the advantage of `add_subdirectory`, in this case, is that when CMake is processing the file, it will process it where it currently lives in the folder structure. This means that `${CMAKE_CURRENT_SOURCE_DIR}` will refer to `path/to/ch8/part-2/app/third-party`, not `path/to/ch8/part-2/app`.
			If we’d instead used `include` and referred to the `CMakeLists.txt` file explicitly (`include(third-party/CMakeLists.txt)`), this would have the effect of copying the contents of the file directly to where the `include` call is (we’re effectively substituting the `include` call with the contents of the file). The issue with this is that `${CMAKE_CURRENT_SOURCE_DIR}`, which is used inside `ch8/part-2/app/third-party/CMakeLists.txt`, will refer to `ch8/part-2/app` instead of `ch8/part-2/app/third-party`. This means that the behavior of our third-party `ExternalProject_Add` commands will differ when called as part of a super build instead of when being invoked directly from the `third-party` folder. In this instance, the relative paths we’re using to refer to our internal library files will not resolve correctly and the location of the third-party install folder will be in `app` instead of `app/third-party`. We want to avoid this, so `add_subdirectory` is a better choice in this instance.
			Let’s now follow the execution flow CMake will take and look at the changes made to our third-party `CMakeLists.txt` file found in `ch8/part-2/app/third-party`. The convenient thing is that we’ve kept the ability to build the third-party dependencies separately if we want to. Things will work just as they did before when we’re not using a super build.
			The first change is to configure where the third-party build artifacts should go depending on whether we’re using a super build or not. If we’re using a super build, we want to use a separate build folder to store all the third-party dependency files. The reason for this is to keep the same separation we had before when building our third-party dependencies separately. If we don’t do this, our third-party build artifacts will be added to the same build folder as our main application. This means that if we want to remove our application build folder and rebuild, we need to build all our third-party dependencies again.
			One way to achieve this is with the following check:

```

if(SUPERBUILD AND NOT PROJECT_IS_TOP_LEVEL)

set(PREFIX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build)

else()

set(PREFIX_DIR ${CMAKE_CURRENT_BINARY_DIR})

endif()

```cpp

			When using a super build, the build files will be added to `third-party/build` (`CMAKE_CURRENT_SOURCE_DIR` will refer to the folder that the current `CMakeLists.txt` file is being processed in). The one downside to this approach is that we’ve hard-coded where the third-party build folder is, and it cannot be modified by users (they could still build the third-party libraries separately and specify `CMAKE_CURRENT_BINARY_DIR` using `-B`, but not as part of a super build).
			To make things more flexible, we can provide a CMake cache variable with a reasonable default that users can override:

```

set(THIRD_PARTY_BINARY_DIR

"${CMAKE_SOURCE_DIR}/build-third-party"

CACHE STRING "第三方构建文件夹")

if(NOT IS_ABSOLUTE ${THIRD_PARTY_BINARY_DIR})

set(THIRD_PARTY_BINARY_DIR

"${CMAKE_SOURCE_DIR}/${THIRD_PARTY_BINARY_DIR}")

endif()

set(PREFIX_DIR ${THIRD_PARTY_BINARY_DIR})

```cpp

			Here, we introduce `THIRD_PARTY_BINARY_DIR`, which we default to `app/build-third-party` (we could have stuck with `third-party/build`, but this way, our app and third-party build folders will stay closer together to make clean-up easier). We also ensure to handle if a user provides a relative path by using `if(NOT IS_ABSOLUTE ${THIRD_PARTY_BINARY_DIR})`. In this case we append the path provided to `CMAKE_SOURCE_DIR` (which, in our case, will be the `app` folder). This check also treats paths beginning with `~/` on macOS and Linux as absolute paths. It’s a little more code, but the increased flexibility can be incredibly useful for our users.
			The extra check of `AND NOT PROJECT_IS_TOP_LEVEL` is to guard against someone accidentally setting `SUPERBUILD` to `ON` when building the third-party dependencies separately as their own project. `SUPERBUILD` will have no effect if this is the case. `PROJECT_IS_TOP_LEVEL` can be used to check whether the preceding call to `project` was from the top-level `CMakeList.txt` file or not (for more information, please see [`cmake.org/cmake/help/latest/variable/PROJECT_IS_TOP_LEVEL.html`](https://cmake.org/cmake/help/latest/variable/PROJECT_IS_TOP_LEVEL.html)). 
			By introducing the `PREFIX_DIR` variable, we can later pass this to `PREFIX` in the `ExternalProject_Add` command, along with the dependency name, to ensure that the build files wind up in `app/build-third-party/<dep>` instead of `app/build`. When building normally, `CMAKE_CURRENT_BINARY_DIR` will resolve to whatever the user sets as their build folder as part of the third-party CMake configure command.
			We use `PREFIX_DIR` in the `ExternalProject_Add` command like so:

```

ExternalProject_Add(

<name>

...

PREFIX ${PREFIX_DIR}/<name>

BINARY_DIR ${PREFIX_DIR}/<name>/build/${build_type_dir}

INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install

...)

```cpp

			The `PREFIX` argument to `ExternalProject_Add` sets the root directory for the dependency. We use the new `PREFIX_DIR` variable, along with the external project name. This ensures that all dependencies we’re building are isolated from one another, avoiding any risk of file naming collisions when downloading and building them. It also makes it easier to rebuild a specific dependency by deleting one of the subfolders and then running the CMake configure and build commands again. Lastly, we will also update `BINARY_DIR` to refer to `PREFIX_DIR` instead of `CMAKE_CURRENT_BINARY_DIR` to handle whether we’re building things as a super build or not.
			Each dependency has the same changes applied; there’s only one more change at the end of the file, also wrapped inside a super build check. The change is yet another call to `ExternalProject_Add`, only this time with a bit of a twist: we’re calling it with the source directory of our top-level CMake project:

```

if(SUPERBUILD AND NOT PROJECT_IS_TOP_LEVEL)

ExternalProject_Add(

${CMAKE_PROJECT_NAME}_superbuild

DEPENDS SDL2 bgfx mc-gol mc-draw

SOURCE_DIR ${CMAKE_SOURCE_DIR}

BINARY_DIR ${CMAKE_BINARY_DIR}

CMAKE_ARGS -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}

-DSUPERBUILD=OFF ${build_type_arg}

INSTALL_COMMAND "")

endif()

```cpp

			We start by naming the external project the same as the CMake project currently being processed, with `_superbuild` appended to differentiate them (`CMAKE_PROJECT_NAME` refers to the top-level CMake project). We then ensure that the project depends on all our third-party dependencies using the `DEPENDS` argument so it will only be built when they all are ready. `SOURCE_DIR` is where we set `ExternalProject_Add` to look for our root `CMakeLists.txt` file (`CMAKE_SOURCE_DIR` refers to the top level of the current CMake source tree, which is usually synonymous with the folder containing our root `CMakeLists.txt` file). We also let it know where to find the third-party dependencies with `CMAKE_PREFIX_PATH`.
			We next pass through the `SUPERBUILD` option, only this time, we’re explicitly setting it to `OFF`. The key insight to this approach is realizing that we’re calling our root `CMakeLists.txt` script recursively. The second time through, as `SUPERBUILD` is `OFF`, we’ll process the file as normal after waiting for all our dependencies to become available (this is what `DEPENDS` guarantees). Finally, we need to disable the `INSTALL_COMMAND` option, as our application doesn’t currently provide an install target (we haven’t added any install functionality), so we just set it to an empty string.
			Looping back to the top-level `CMakeLists.txt` file in `ch8/part-2/app`, after the call to `add_subdirectory`, we simply `return` and finish processing (remember, we will have processed this file in its entirety in the `ExternalProject_Add` command with `SUPERBUILD` set to `OFF`).
			Those are all the changes we need to support super builds. To make enabling them a bit easier, we’ve also added a new CMake configure preset called `multi-ninja-super`. It uses a new hidden configure preset called `super`, as shown here, with `SUPERBUILD` set to `ON`:

```

"name": "super",

"hidden": true,

"cacheVariables": {

"SUPERBUILD": "ON"

}

```cpp

			The new preset we added then inherits `super` as well as `multi-ninja`:

```

"name": "multi-ninja-super",

"inherits": ["multi-ninja", "super"]

```cpp

			To take advantage of this, from `ch8/part-2/app`, run the following commands:

```

cmake --preset multi-ninja-super

cmake --build build/multi-ninja-super

```cpp

			Taking it one step further, we can also add a build preset that uses the new configure preset, and finally a workflow preset that uses them both together:

```

# 构建预设

"name": "multi-ninja-super",

"configurePreset": "multi-ninja-super"

# 工作流预设

"name": "multi-ninja-super",

"steps": [

{

"type": "configure",

"name": "multi-ninja-super"

},

{

"type": "build",

"name": "multi-ninja-super"

}

]

```cpp

			This allows us to configure and build everything in one command:

```

cmake --workflow --preset multi-ninja-super

```cpp

			Ninja and super builds on Windows
			It is possible, if you are using super builds on Windows, that after the first build, you may hit the `ninja: error: failed recompaction: Permission denied` error. This appears to be a Windows-specific issue with Ninja related to file paths. Running the same CMake command again will resolve the error. However, if the issue persists, it may be worth experimenting with other generators on Windows such as Visual Studio.
			One last reminder is that, by default, this will build in a debug configuration (`Debug`), which we might not want, so adding `"configuration": "Release"` to our build preset is likely a good idea (see `ch8/part-2/app/CMakePreset.json` for a full example).
			We can then run our app as usual from the project root directory (`ch8/part-2/app`) with the following command (not forgetting to first build the shaders):

```

./compile-shaders-<platform>.sh/bat

./build/multi-ninja-super/Release/minimal-cmake_game-of-life_window

```cpp

			Earlier in the section, we briefly touched on the fact that super builds and regular builds can coexist seamlessly. For example, we can configure our existing `multi-ninja` preset after configuring and building the super build, as we know all third-party dependencies are now downloaded and installed (`multi-ninja` still has `CMAKE_PREFIX_PATH` set to `app/third-party/install`). This can be quite useful as configuring `multi-ninja-super` again and then building the application will trigger a check of all dependencies. This is usually quite fast, but if the dependencies are stable and aren’t changing often, creating a separate non-super build folder for active development will avoid this. You essentially have two build folders underneath `build`:

```

build/multi-ninja-super # 超级构建

build/multi-ninja # 常规构建

```cpp

			We get these two build folders thanks to our use of CMake presets and the `binaryDir` property, which lets us use the current preset name (`${sourceDir}/build/${presetName}`). *Chapter 5*, *Streamlining CMake Configuration*, contains more information about this topic for reference.
			One final gotcha to mention is that when using a multi-config generator, changing the config passed to the build command will only trigger the application to rebuild in the new configuration, not all the dependencies. To ensure that all dependencies are rebuilt, it is necessary to configure before running the build command (`cmake --``build <build-folder>`).
			For example, this might look as follows:

```

# 默认构建为 Debug

cmake --preset multi-ninja-super

# 在 Debug 模式下构建所有内容

cmake --build build/multi-ninja-super

# 仅在 Release 模式下构建应用程序

cmake --build build/multi-ninja-super --config Release

# 必须先重新运行配置

cmake --preset multi-ninja-super

# 现在以 Release 模式构建所有内容

cmake --build build/multi-ninja-super --config Release

```cpp

			Using a single config generator behaves the same. It’s just something to be aware of, as normally, changing the `--config` option passed to the CMake build command with multi-config generators will rebuild everything in the new configuration (this is, unfortunately, one downside of using the super build pattern, but it’s a fairly minor one given the benefits it brings).
			The last thing to be aware of with super builds is clean-up. Before, we only needed to delete the build folder to remove all build artifacts, but now, because we’ve split things, there are two folders (three, if you count the install folder) to delete. For completeness, to get back to a clean state, remember to delete the following folders (default locations listed here):

```

app/build

app/build-third-party

app/third-party/install

```cpp

			With super builds, we can now run a single CMake command and have our project downloaded and built in one step. This is a substantial improvement on what we had before, but there’s one last issue to address: automating the shader compilation step. We’ll look at how to achieve this in the next section.
			Automating scripts with CMake
			We’ve removed a lot of manual steps that we were dealing with at the start of the chapter, but one remains. This is the requirement to build the shaders needed by bgfx to transform and color our geometry. Up until now, we’ve been relying on running custom `.bat`/`.sh` scripts from the `app` folder before running our *Game of Life* application, but there’s a better possibility. In this section, we’ll show how to make this process part of the build itself, and use CMake to achieve a cross-platform solution without the need for OS-specific scripts.
			To start with, we’re going to do away with our existing `.bat`/`.sh` scripts and replace them with `.cmake` files. We’ll pick macOS as the first platform to update; the file will be called `compile-shader-macos.cmake`, and will live under a new `cmake` folder in the `app` directory (equivalent files for Windows and Linux will differ in the exact same way as the existing scripts).
			We’re eventually going to invoke these scripts from our top-level `CMakeLists.txt` file. However, before we do, it’s useful to introduce a CMake operation we haven’t covered so far, and that is the ability to run a CMake script from the command line using `cmake –P` (see [`cmake.org/cmake/help/latest/manual/cmake.1.html#run-a-script`](https://cmake.org/cmake/help/latest/manual/cmake.1.html#run-a-script) for more details). As a quick example, we can create a file called `hello-world.cmake` and add a simple `message` command to output `Hello, world!`:

```

# hello-world.cmake

message(STATUS "Hello, World!")

```cpp

			If we invoke it from the command line by running `cmake -P hello-world.cmake`, we’ll see the following output:

```

-- Hello, World!

```cpp

			(If we include `hello-world.cmake` in a `CMakeLists.txt` file, it will run at configure time and `Hello, World!` will be printed then).
			The CMake functionality to invoke a script also supports first providing CMake variables from the command line using the familiar `-D` argument introduced in *Chapter 2*, *Hello, CMake!* (importantly appearing before `-P`):

```

cmake -DA_USEFUL_SETTING=ON -P cmake-script.cmake

```cpp

			We’ll use this in our shader script example a little later to help control the output when invoking the command.
			CMake provides a wealth of useful modules and functions to support file and path manipulation. We’re going to take advantage of them as we craft a CMake script to build our shaders. It’s important to ensure that we have a consistent working directory when invoking `compile-shader-<platform>.cmake`. There are some subtle differences when running from a top-level CMake project and invoking the script using `-P` directly. For example, if we decided to use `CMAKE_SOURCE_DIR` when specifying our paths, this would work correctly when running from the top-level `CMakeLists.txt` file, and when invoking `compile-shader-<platform>.cmake` from the app folder (e.g., `cmake -P cmake/compile-shader-macos.cmake`), but would fail if a user tried to run it from the nested `cmake` folder itself. This is because `CMAKE_SOURCE_DIR` will default to the folder holding the top-level `CMakeLists.txt` file when part of a CMake configure step, and to the folder CMake was invoked from when running `cmake -P path/to/cmake-script.cmake` (this is the same problem we had with the `.``sh`/`.bat` scripts).
			To account for these differences, we’re going to use a CMake path-related function to ensure that our script’s working directory is always set to the `app` folder. The function we’re going to use is called `cmake_path`. Added in CMake `3.20`, `cmake_path` provides utilities to manipulate paths, decoupled from the filesystem itself (to learn more about `cmake_path`, see [`cmake.org/cmake/help/latest/command/cmake_path.html`](https://cmake.org/cmake/help/latest/command/cmake_path.html)). In our case, we’d like to find the directory containing our `compile-shader-<platform>.cmake` file. This can be performed with the following command:

```

cmake_path(

GET CMAKE_SCRIPT_MODE_FILE PARENT_PATH

COMPILE_SHADER_DIR)

```cpp

			In the preceding command, we can see the following arguments:

				*   The first argument, `GET`, describes the type of operation we’d like to perform.
				*   The next argument, `CMAKE_SCRIPT_MODE_FILE` ([`cmake.org/cmake/help/latest/variable/CMAKE_SCRIPT_MODE_FILE.html`](https://cmake.org/cmake/help/latest/variable/CMAKE_SCRIPT_MODE_FILE.html)), holds the full path to the current script being processed. It’s important to note that this variable is only set when using `cmake -P` to execute the script. It will not be populated when using `include`. A check for this variable can be included at the top of the script and a warning issued if a user incorrectly tries to include it (see `ch8/part-3/app/cmake/compile-shader-<platform>.cmake` for an example).
				*   The following argument, `PARENT_PATH`, is the component to retrieve from the preceding path. In this case, we are requesting the parent path of the current script file (essentially, the directory it is in). To see what other components are available, please see [`cmake.org/cmake/help/latest/command/cmake_path.html#decomposition`](https://cmake.org/cmake/help/latest/command/cmake_path.html#decomposition).
				*   The final argument, `COMPILE_SHADER_DIR`, is the variable to populate the result with.

			Now we have this directory, we just need to go one level up to reach the `app` folder. We can achieve this using the same command, only substituting the first argument with the variable we populated in the preceding command.

```

cmake_path(

GET COMPILE_SHADER_DIR PARENT_PATH

COMPILE_SHADER_WORKING_DIR)

```cpp

			We now have a consistent and portable way to automatically retrieve the `app` folder. We can use the `COMPILE_SHADER_WORKING_DIR` variable in the following CMake script commands.
			CMake provides another useful utility called `file` that can be used for a wide array of file and path manipulations (as opposed to `cmake_path`, this command does interact with the filesystem). In our simple case, we just need to create a new folder (the `build` folder in the `app/shader` directory), which can be achieved with the following `file` command:

```

file(

MAKE_DIRECTORY

${COMPILE_SHADER_WORKING_DIR}/shader/build)

```cpp

			The first argument is the operation to perform, and the second is where to do it. This ensures that we now have an output directory to hold our compiled shader files. To learn more about the `file` command, see [`cmake.org/cmake/help/latest/command/file.html`](https://cmake.org/cmake/help/latest/command/file.html).
			We’re next going to make use of a CMake command called `execute_process`, which allows us to run child processes from within a CMake script. In this case, we’re going to replicate the contents of our `compile_shader_macos.sh` file inside the `execute_process` command. The following is an example of what this looks like:

```

execute_process(

COMMAND

third-party/install/bin/shaderc

-f shader/vs_vertcol.sc

-o shader/build/vs_vertcol.bin

--platform osx --type vertex

-i ./ -p metal --verbose

WORKING_DIRECTORY ${COMPILE_SHADER_WORKING_DIR})

```cpp

			We first call `execute_process`, and then pass the `COMMAND` argument. What follows are the same instructions we would pass at the command line, which were previously invoked from our `.sh` script. We then pass one more argument, `WORKING_DIRECTORY`, to specify where the listed commands should be run relative to (this is populated by the variable we created earlier referring to the `app` directory, regardless of whether the script is being run using `cmake -P` or whether it is being invoked from a `CMakeLists.txt` file). We can now build our shaders using `cmake -P path/to/app/cmake/compile-shader-macos.cmake` from any folder of our choosing (to understand what else `execute_process` can do, see [`cmake.org/cmake/help/latest/command/execute_process.html`](https://cmake.org/cmake/help/latest/command/execute_process.html)).
			Before we look at invoking our new scripts from `CMakeLists.txt` as part of the main build, there’s a small improvement we can make to our new `compile-shader-macos.cmake` file. Up until now, we’ve been passing the `--verbose` flag to the *bgfx* `shaderc` program to show the full output of compiling our shaders. This can sometimes be useful, but it’s unlikely that we want to see this as part of the main build every time we either configure or build using CMake. Even with the `--verbose` argument removed, the output is still generated when invoking `shaderc`, which, in the default case, we might want to hide.
			To work around this, let’s introduce a new CMake variable called `USE_VERBOSE_SHADER_OUTPUT` to our `compile-shader-<platform>.cmake` scripts. This will default to `OFF` and will control two internal CMake variables. The first is `VERBOSE_SHADER_OUTPUT`, which will substitute the direct reference to `--verbose`:

```

option(

USE_VERBOSE_SHADER_OUTPUT

"显示着色器编译输出" OFF)

如果(USE_VERBOSE_SHADER_OUTPUT)

set(VERBOSE_SHADER_OUTPUT --verbose)

endif()

execute_process(

COMMAND

...

${VERBOSE_SHADER_OUTPUT}

WORKING_DIRECTORY ${COMPILE_SHADER_WORKING_DIR})

```cpp

			When we invoke `cmake -P cmake/compile-shader-<platform>.cmake`, we now won’t, by default, see the full output from `shaderc`, but we can easily enable it again by setting `VERBOSE_SHADER_OUTPUT` to `ON`:

```

cmake --verbose，shaderc 仍然会将一些信息输出到终端，这可能会干扰正常的 CMake 构建输出。为了隐藏这些信息，我们可以引入另一个 CMake 变量叫做 QUIET_SHADER_OUTPUT，然后将其设置为 ERROR_QUIET（或在 Linux 上设置为 OUTPUT_QUIET）以抑制`execute_process`命令的所有输出（OUTPUT_QUIET 和 ERROR_QUIET 分别对应标准输出和标准错误输出，例如 C 语言中的`fprintf`，`stdout`和`stderr`，以及 C++中的`std::cout`和`std::cerr`）。

            我们的最终代码如下：

```cpp
if(USE_VERBOSE_SHADER_OUTPUT)
  set(VERBOSE_SHADER_OUTPUT --verbose)
else()
  set(QUIET_SHADER_OUTPUT ERROR_QUIET OUTPUT_QUIET)
endif()
execute_process(
  COMMAND
  ...
  ${VERBOSE_SHADER_OUTPUT}
  ${QUIET_SHADER_OUTPUT}
  WORKING_DIRECTORY ${COMPILE_SHADER_WORKING_DIR})
```

            这意味着我们目前无法拥有非详细输出；要么全开，要么全关，但这通常足以满足我们调用这些脚本的需求。所有`compile-shader-<platform>.cmake`文件中的变化几乎是相同的，现在我们已经准备好查看如何从我们应用的`CMakeLists.txt`文件中调用这些脚本。

            从 CMakeLists.txt 调用 CMake 脚本

            我们的脚本现在可以从`CMakeLists.txt`文件中调用。首先，我们需要根据构建的平台引用正确的文件。我们可以通过简单的条件检查来实现：

```cpp
if(WIN32)
  set(COMPILE_SHADER_SCRIPT
      ${CMAKE_SOURCE_DIR}/cmake/compile-shader-windows.cmake)
elseif(LINUX)
  set(COMPILE_SHADER_SCRIPT
      ${CMAKE_SOURCE_DIR}/cmake/compile-shader-linux.cmake)
elseif(APPLE)
  set(COMPILE_SHADER_SCRIPT
      ${CMAKE_SOURCE_DIR}/cmake/compile-shader-macos.cmake)
endif()
```

            现在我们可以引用`COMPILE_SHADER_SCRIPT`来获取适合我们平台的文件。接下来有两种不同的方式可以自动调用我们的脚本。一个方法是使用`include`将脚本直接引入到我们的`CMakeLists.txt`文件中：

```cpp
include(${COMPILE_SHADER_SCRIPT})
```

            不幸的是，现有的`compile-shader-<platform>.cmake`文件不能直接与此方法一起使用。我们需要更新如何填充`COMPILE_SHADER_WORKING_DIR`。我们可以通过以下检查来实现：

```cpp
if(CMAKE_SCRIPT_MODE_FILE AND NOT CMAKE_PARENT_LIST_FILE)
  # existing approach
else()
  set(COMPILE_SHADER_WORKING_DIR ${CMAKE_SOURCE_DIR})
endif()
```

            当 CMake 脚本作为`CMakeLists.txt`文件的一部分被调用时，`CMAKE_SCRIPT_MODE_FILE`不会被设置（也叫填充），而`CMAKE_PARENT_LIST_FILE`是包含它的 CMake 文件的完整路径。通过使用这两个检查，我们可以确保只有在文件以脚本模式运行且未被其他文件包含时，才会执行第一个分支。如果我们知道该文件是从`CMakeLists.txt`文件中调用的，我们可以简单地将`COMPILE_SHADER_WORKING_DIR`设置为`CMAKE_SOURCE_DIR`（它将是包含根`CMakeLists.txt`文件的文件夹），这样一切就会按预期工作。

            使用这种方法时，每次配置时都会构建着色器。还有一种替代方法可以代替使用`include`，那就是使用我们之前遇到过的 CMake 命令`add_custom_command`。通过`add_custom_command`，我们可以指定一个目标和命令执行的时机（在下面的示例中，我们使用`POST_BUILD`在应用程序构建完成后调用该命令）。完整的命令如下：

```cpp
add_custom_command(
  TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -P ${COMPILE_SHADER_SCRIPT}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  VERBATIM)
```

            这个命令对我们的用户非常方便，并确保在应用程序运行之前先编译着色器。缺点是，目前该命令可能比严格必要时运行得更频繁，如果命令变得更复杂并开始需要更长时间运行，将来可能会成为一个问题。

            还有一个`add_custom_command`的替代版本，它不是接受一个目标（`TARGET`），而是接受一个输出文件（`OUTPUT`）。通过`DEPENDS`参数可以列出依赖项，只有在输出文件需要更新时，命令才会执行。这种方法非常高效，但遗憾的是，设置起来稍微复杂一些，而由于前面的命令执行较快，当前使用的是较简单的版本（要了解更多关于`add_custom_command`的信息，请查看[`cmake.org/cmake/help/latest/command/add_custom_command.html`](https://cmake.org/cmake/help/latest/command/add_custom_command.html)））。

            添加新命令后，我们已经拥有了使用一个命令构建整个应用程序和附带资源（着色器）所需的一切。从`ch8/part-3/app`目录下，运行以下命令：

```cpp
cmake --workflow --preset multi-ninja-super
```

            一旦构建完成，剩下的就是运行应用程序本身：

```cpp
./build/multi-ninja-super/Release/minimal-cmake_game-of-life_window
```

            当然，我们也可以像之前一样使用 CMake 的`--preset`和`--build`参数分别进行配置和构建。不过，利用`--workflow`在这里特别方便。

            审查`ch8/part-3/app/CMakeLists.txt`和`ch8/part-3/app/cmake/compile-shader-<platform>.cmake`文件，查看上下文中的所有内容。你可能会注意到一个小变化，那就是在每个`compile-shader-<platform>.cmake`文件的顶部，加入了一个简化的检查，以确保它们必须在脚本模式下运行：

```cpp
if(NOT CMAKE_SCRIPT_MODE_FILE)
  message(
    WARNING
      "This script cannot be included, it must be executed using `cmake -P`")
  return()
endif()
```

            提醒一下，`ch8/part-2/app`和`ch8/part-3/app`。

            在嵌套文件中设置选项

            我们为简化和精简应用程序构建所采取的步骤已经带来了巨大的变化，并将在未来节省时间和精力。然而，我们在这个过程中不幸失去了一样东西，那就是调整依赖项构建方式的能力。之前，当我们使用`FetchContent`并直接构建依赖项时，我们可以传递各种构建选项来设置是否将特定库构建为静态库或共享库。在*第七章*中，*为库添加安装支持*，当我们考虑单独构建库并手动安装时，我们也可以决定如何构建它们。不幸的是，通过使用`ExternalProject_Add`，我们失去了一些灵活性，因为没有额外的支撑框架，无法直接将选项传递给`ExternalProject_Add`命令。

            幸运的是，失去的灵活性并不难恢复。所需的仅仅是创建我们自己的 CMake 选项，然后将它们作为`CMAKE_ARGS`参数的一部分转发给内部的`ExternalProject_Add`命令。

            例如，如果我们查看`ch8/part-4/app/third-party/CMakeLists.txt`，我们可以看到在文件顶部，我们添加了两个新选项：

```cpp
option(
  MC_GOL_SHARED
  "Enable shared library for Game of Life" OFF)
option(
  MC_DRAW_SHARED "Enable shared library for Draw" OFF)
```

            我们使用了与库中实际存在的名称相同的名称，以保持一致性，但如果我们选择将变量与我们正在构建的应用程序分组，仍然可以自由调整命名。然后，我们将这些新值传递给`ExternalProject_Add`，其形式如下：

```cpp
ExternalProject_Add(
  mc-draw
  ...
  CMAKE_ARGS ... -DMC_DRAW_SHARED=${MC_DRAW_SHARED}
  ...)
```

            这使我们即使在使用`ExternalProject_Add`引入依赖项时，也能决定是否将其构建为静态库或共享库。我们不希望更改的库暴露的其他选项可以硬编码。

            我们还需要对`compile-shader-<platform>.cmake`脚本做同样的事情。由于我们从`CMakeLists.txt`文件调用脚本的方式，`USE_VERBOSE_SHADER_OUTPUT`设置不会自动检测到（如果我们使用`include`，它会被拾取并添加到主项目的`CMakeCache.txt`文件中）。为了解决这个问题，我们只需将该设置添加到`CMakeLists.txt`文件中，然后将其传递给脚本的调用：

```cpp
option(
  USE_VERBOSE_SHADER_OUTPUT 
  "Show output from shader compilation" OFF)
...
add_custom_command(
  TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -D USE_VERBOSE_SHADER_OUTPUT=${USE_VERBOSE_SHADER_OUTPUT}
    -P ${COMPILE_SHADER_SCRIPT}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  VERBATIM)
```

            请参阅`ch8/part-4/app/CMakeLists.txt`以获取完整示例。

            这种变量传递的需求主要出现在`ExternalProject_Add`中。当使用超级构建时，我们需要记住遵循相同的方法，将选项传递到嵌套的应用程序中。这就是为什么有时使用普通的非超级构建项目会很有用（请参见`ch8/part-4/app/CMakePresets.json`中的`multi-ninja`和`multi-ninja-super` CMake 预设作为示例）。配置应用程序的选项数量通常较少，您可以将剩余的、不需要更改的选项直接设置在`ExternalProject_Add`调用中，但有时提供一种更改这些选项的方法会很有用。

            安装应用程序

            本章的最后，我们将看一个最终的添加内容，那就是如何为我们的应用程序添加安装支持。这有助于为打包做好准备，并确保我们的应用程序具有可移植性。

            我们要做的第一个更改是将一个`CMAKE_INSTALL_PREFIX`变量添加到我们应用程序的`CMakePresets.json`文件中，以确保我们的应用程序安装在相对于项目的路径中：

```cpp
"CMAKE_INSTALL_PREFIX": "${sourceDir}/install"
```

            接下来的一些更改将专门针对`ch8/part-5/app/CMakeLists.txt`。首先，我们需要像为库一样包含`GNUInstallDirs`，以访问标准的 CMake 安装位置（在这个例子中，我们只关心`CMAKE_INSTALL_BINDIR`）。

            我们想要实现的高层目标是拥有一个可重定位的文件夹，包含我们的应用程序可执行文件、需要由应用程序加载的共享库以及运行时所需的资源（我们编译的着色器文件）。我们可以通过以下 CMake 安装命令来实现这一目标。

            第一个步骤很简单，它将应用程序的可执行文件复制到`install`文件夹：

```cpp
install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
```

            我们提供目标以复制，并使用`RUNTIME`类型来指代可执行文件，同时指定复制目标路径（这通常是`bin`，并且会相对于我们在`CMakePresets.json`文件中提供的`CMAKE_INSTALL_PREFIX`变量）。

            接下来，我们需要复制应用程序启动时需要加载的共享库文件。由于我们正在开发一个跨平台应用程序，为了简化起见，我们将所有共享库文件（Windows 上的`.dll`、macOS 上的`.dylib`和 Linux 上的`.so`）复制到与应用程序相同的文件夹。这与我们之前在 Windows 上做的非常相似，但现在我们将在所有平台上做同样的事，以保持一致性。复制这些文件的简化安装命令如下所示：

```cpp
install(
  FILES
    $<TARGET_FILE:SDL2::SDL2>
    ...
```

            由于我们的*生命游戏*和*绘图*库可以编译为静态库或共享（动态）库，我们需要检查目标是否是正确的类型，然后再复制它，否则我们将不必要地复制`.lib`/`.a`静态库文件。

            我们可以通过这里显示的生成器表达式来实现：

```cpp
$<$<STREQUAL:$<TARGET_PROPERTY:minimal-cmake::line,TYPE is equal to SHARED_LIBRARY, then substitute the path to the shared library, otherwise, do nothing (the expression will evaluate to an empty string).
			Sticking with the dynamic libraries, there’s another minor change we need to make. If you recall *Chapter 4*, *Creating Libraries for FetchContent*, we discussed the topic of making libraries relocatable on macOS and Linux by changing the `RPATH` variable of the executable. We achieved this by using `set_target_properties` to update the `BUILD_RPATH` property of the executable. To ensure things work correctly for both build and install targets, we need to update this command slightly. The changes are shown here:

```

set_target_properties(

${PROJECT_NAME}

PROPERTIES

INSTALL_RPATH

"$<$<PLATFORM_ID:Linux>:$ORIGIN>$<$<PLATFORM_ID:Darwin>:@loader_path>"

对于 BUILD_RPATH 属性，我们将 INSTALL_RPATH 属性更新为在 Linux 上解析为 $ORIGIN，在 macOS 上解析为 @loader_path（这样做是为了使可执行文件在与自身相同的文件夹中查找共享库）。由于我们希望正常构建的目标和已安装的目标表现一致，因此我们还将 BUILD_WITH_INSTALL_RPATH 设置为 TRUE，这大致相当于将相同的生成器表达式传递给 BUILD_RPATH，就像之前一样（我们在这里的意思是 BUILD_RPATH 应该与 INSTALL_RPATH 相同）。

            现在，通过将共享库分别复制到构建和安装文件夹中，我们实现了构建目标和安装目标之间的相同行为。构建完成后，我们可以安全地移除已安装的依赖项（即在 `app/third-party/install` 中的已安装库文件），并继续运行应用程序（然而，这样会破坏我们重新编译和构建的能力，因为在没有先通过生成新的超级构建或配置并从 `third-party` 文件夹构建的情况下，无法恢复第三方依赖项）。

            `RPATH` 处理是一个复杂的话题，这里提供的解决方案只是处理共享库安装的一种方法。要了解更多内容，请参考 CMake 属性文档页面上的与 `RPATH` 相关的变量（[`cmake.org/cmake/help/latest/manual/cmake-properties.7.html`](https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html)）以及 CMake 社区 Wiki 上关于 `RPATH` 处理的部分（[`gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling`](https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling)）。

            为了确保完全的跨平台兼容性，还需要进行最后一个添加，才能正确处理在 Linux 上为我们的应用程序安装 SDL 2 依赖项。当 SDL 2 在 Linux 上构建时，它会提供作为共享库的一部分的多个文件。这些文件与 `libSDL2-2.0.so.0.3000.2` 相关，但这不是动态链接器查找的文件。`libSDL2-2.0.so.0` 文件是指向 `libSDL2-2.0.so.0.3000.2` 的符号链接，它是动态链接器在查找 SDL 2 库时使用的文件。确保我们安装这两个文件非常重要；否则，应用程序将在运行时找不到共享库。

            为了支持这一点，我们只需要在 `add_custom_command` 和 `install` 调用中再添加一条规则，那就是除了 `TARGET_FILE` 之外，还需要安装 `TARGET_NAME_SOFILE`，如下所示：

```cpp
$<TARGET_FILE:SDL2::SDL2>
$<$<PLATFORM_ID:Linux>:$<TARGET_SONAME_FILE:SDL2::SDL2>>
```

            我们还添加了一个条件生成器表达式，只有在 Linux 平台上才会评估此表达式，因为在其他平台上不需要。

            最后，我们需要安装的最后一组文件是我们编译的着色器。我们将它们安装到与从 `app` 文件夹启动时相同的相对位置。实现这一目标的 CMake `install` 命令如下所示：

```cpp
install(DIRECTORY ${CMAKE_SOURCE_DIR}/shader/build
        DESTINATION ${CMAKE_INSTALL_BINDIR}/shader)
```

            我们将`shader`下的`build`目录复制到我们安装可执行文件和共享库文件的相同文件夹中。这样，当我们从`ch8/part-5/app/install/bin`运行应用程序时，它在相对位置上看起来与之前从`ch8/part-5/app`运行时相同。

            经过最后的更改，我们拥有了安装应用程序所需的一切。我们现在只需要运行以下命令，从`ch8/part-5/app`文件夹构建并安装我们的应用程序：

```cpp
cmake --build build/multi-ninja-super
```

            当从超级构建目录（`multi-ninja-super`）构建时，由于我们使用`ExternalProject_Add`来包装我们的项目（在`app/third-party/CMakeLists.txt`的末尾为`${CMAKE_PROJECT_NAME}_superbuild`），安装操作会在运行`cmake --build build/multi-ninja-super`时自动发生（就像我们直接从`third-party`文件夹构建第三方依赖一样）。在配置后首次构建时，无需传递`--target install`（尝试传递该参数实际上会导致错误，因为找不到安装目标）。之后的构建或从普通构建文件夹（例如`build/multi-ninja`）构建时，将需要`--target install`参数，因为安装目标将可用。最后，再次运行配置命令（例如`cmake --preset multi-ninja-super`）将重置此行为，以便用于后续的构建。

            如果我们之前使用`--workflow`预设构建了我们的应用程序，我们也可以改用 CMake 的`--install`命令：

```cpp
cmake --install build/multi-ninja-super
```

            要启动应用程序，安装完成后，切换到`ch8/part-5/app/install/bin`目录并运行`./minimal-cmake_game-of-life_window`。可以自由地探索`ch8/part-5/app`的内容，以查看所有上下文，并通过运行我们到目前为止介绍的不同 CMake 命令进行实验（从`cmake --preset list`开始，然后运行`cmake --preset <preset>`是一个不错的起点）。还可以尝试将`app/install/bin`文件夹复制或移动到新的位置（或与之匹配的操作系统和架构的计算机上），以验证应用程序是否仍然能够启动并成功运行。

            总结

            是时候再休息一下，让我们所涵盖的内容稍微消化一下了。我们触及了一些高级 CMake 特性，如果你感到有些头晕也不用担心。你练习和实验这些概念的次数越多，理解会越来越清晰。

            在本章中，我们从手动安装自己的库转向利用`ExternalProject_Add`来自动化安装过程。这大大减少了设置项目时的繁琐步骤，并且是一种适用于未来项目的有用策略。接着，我们查看了为项目设置超级构建的过程，这提供了一种使用单个命令构建所有内容的方式，同时不失去我们所期望的灵活性。这种技术进一步简化了项目配置，是为用户创建应用程序时提供的极好的默认设置。

            之后，我们了解了 CMake 如何替代跨平台脚本，并自动化外部过程，如着色器编译，将其纳入核心构建，而不是事后考虑的事情。这可以节省很多在创建和支持定制的每个平台脚本时的开销。接下来，我们花了一些时间理解如何暴露嵌套依赖中的自定义点，以继续让用户控制他们如何构建库。没有这一点，用户可能不得不编辑 `CMakeLists.txt` 文件，进而带来另一个维护难题。最后，我们演示了如何安装应用程序，使其共享变得轻松。这使我们更接近一个完全可分发的应用程序，并摆脱了依赖项目布局来运行代码的束缚。

            在下一章，我们将介绍一个与 CMake 一起捆绑的配套工具——CTest。CTest 是一个非常有用的工具，帮助简化执行各种测试。我们将学习如何将测试添加到我们的库和应用程序中，并了解如何使用另一个 CMake 工具——CDash 来共享测试结果。

```cpp

```

```cpp

```
