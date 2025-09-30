# Chapter 23

# Build Systems

For us programmers, building a project and running its various components is the first step in developing a new feature or fixing a reported bug in a project. In fact, this is not limited to C or C++; almost any project with a component written in a compiled programming language, such as C, C++, Java, or Go, needs to be built first.

Therefore, being able to build a software project quickly and easily is a fundamental demand required by almost any party working in the software production pipeline, whether they be developers, testers, integrators, DevOps engineers, or even customer support.

More than that, when you join a team as a newbie, the first thing you do is to build the code base that you are going to work on. Considering all this, then, it's clear that addressing the ability to build a software project is justified, given its importance within the software development process.

Programmers need to build code bases frequently in order to see the results of their changes. Building a project with only a few source files seems to be easy and fast, but when the number of source files grows (and believe me, it happens), building a code base frequently becomes a real obstacle to development tasks. Therefore, a proper mechanism for building a software project is crucial.

People used to write shell scripts to build a huge number of source files. Even though it worked, it took a lot of effort and maintenance to keep the scripts general enough to be used in various software projects. Following that, around 1976 at Bell Labs, the first (or, at least, one of the earliest ones) *build system*, named *Make*, was developed and it was used in internal projects.

After that, Make was used on a massive scale in all C and C++ projects, and even in other projects in which C/C++ were n-ot the main languages.

In this chapter, we are going to talk about widely used *build systems* and *build script generators* for C and C++ projects. As part of this chapter, we will talk about the following topics:

*   First, we will look at what build systems are and what they are good for.
*   Then, we will cover what Make is and how Makefiles should be used.
*   CMake is the next topic. You will read about build script generators and you will learn how to write simple `CMakeLists.txt` files.
*   We'll see what Ninja is and how it is different from Make.
*   The chapter will also explore how CMake should be used to generate Ninja build scripts.
*   We'll delve into what Bazel is and how it should be used. You will learn about `WORKSPACE` and `BUILD` files and how they should be written in a simple use case.
*   Finally, you will be given links to some already-published comparisons of various build systems.

Note that the build tools used in this chapter all need to be installed on your system beforehand. Proper resources and documentation should be available on the internet, since these build tools are being used on a massive scale.

In the first section, we are going to explore what a build system actually is.

# What is a build system?

Put simply, a build system is a set of programs and companion text files that collectively build a software code base. Nowadays, every programming language has its own set of build systems. For instance, in Java, you have *Ant*, *Maven*, *Gradle*, and so on. But what does "building a code base" mean?

Building a code base means producing final products from source files. For example, for a C code base, the final products can be executable files, shared object files, or static libraries, and the goal of a C build system is to produce these products out of the C source files found in the code base. The details of the operations needed for this purpose depend heavily on the programming language or the languages involved in the code base.

Many modern build systems, especially in projects written in a *JVM language* such as Java or Scala, provide an extra service.

They do *dependency management* as well. This means that the build system detects the dependencies of the target code base, and it downloads all of them and uses the downloaded artifacts during the *build process*. This is very handy, especially if there are a great many dependencies in a project, which is usually the case in big code bases.

For instance, *Maven* is one of the most famous building systems for Java projects; it uses XML files and supports dependency management. Unfortunately, we don't have great tools for dependency management in C/C++ projects. Why we haven't got Maven-like build systems for C/C++ projects yet is a matter for debate, but the fact that they have not been developed yet could be a sign that we don't need them.

Another aspect of a build system is the ability to build a huge project with multiple modules inside. Of course, this is possible using shell scripts and writing recursive *Makefiles* that go through any level of modules, but we are talking about the intrinsic support of such a demand. Unfortunately, Make does not offer this intrinsically. Another famous build tool, CMake, does offer that, however. We will talk more about this in the section dedicated to CMake.

As of today, many projects still use Make as their default build system, however, through using CMake. Indeed, this is one of the points that makes CMake very important, and you need to learn it before joining a C/C++ project. Note that CMake is not limited to C and C++ and can be used in projects using various programming languages.

In the following section, we are going to discuss the Make build system and how it builds a project. We will give an example of a multi-module C project and use it throughout this chapter to demonstrate how various build systems can be used to build this project.

# Make

The Make build system uses Makefiles. A Makefile is a text file with the name "Makefile" (exactly this and without any extension) in a source directory, and it contains *build targets* and commands that tell Make how to build the current code base.

Let's start with a simple multi-module C project and equip it with Make. The following shell box shows the files and directories found in the project. As you can see, it has one module named `calc`, and another module named `exec` is using it.

The output of the `calc` module would be a static object library, and the output of the `exec` module is an executable file:

```cpp
$ tree ex23_1
ex23_1/
├── calc
│   ├── add.c
│   ├── calc.h
│   ├── multiply.c
│   └── subtract.c
└── exec
    └── main.c
2 directories, 5 files 
$
```

Shell Box 23-1: The files and directories found in the target project

If we want to build the above project without using a build system, we must run the following commands in order to build its products. Note that we have used Linux as the target platform for this project:

```cpp
$ mkdir -p out
$ gcc -c calc/add.c -o out/add.o
$ gcc -c calc/multiply.c -o out/multiply.o
$ gcc -c calc/subtract.c -o out/subtract.o
$ ar rcs out/libcalc.a out/add.o out/multiply.o out/subtract.o
$ gcc -c -Icalc exec/main.c -o out/main.o
$ gcc -Lout out/main.o -lcalc -o out/ex23_1.out
$
```

Shell Box 23-2: Building the target project

As you can see, the project has two artifacts: a static library, `libcalc.a`, and an executable file, `ex23_1.out`. If you don't know how to compile a C project, or the preceding commands are strange to you, please read *Chapter 2*, *Compilation and Linking*, and *Chapter 3*, *Object Files*.

The first command in *Shell Box 23-2* creates a directory named out. This directory is supposed to contain all the relocatable object files and the final products.

Following that, the next three commands use `gcc` to compile the source files in the `calc` directory and produce their corresponding relocatable object files. Then, these object files are used in the fifth command to produce the static library `libcalc.a`.

Finally, the last two commands compile the file `main.c` from the exec directory and finally link it together with `libcalc.a` to produce the final executable file, `ex23_1.out`. Note that all these files are put inside the out directory.

The preceding commands can grow as the number of source files grows. We could maintain the preceding commands in a shell script file called a *build script*, but there are some aspects that we should think about beforehand:

*   Are we going to run the same commands on all platforms? There are some details that differ in various compilers and environments; therefore, the commands might vary from one system to another. In the simplest scenario, we should maintain different shell scripts for different platforms. Then, it effectively means that our script is not *portable*.
*   What happens when a new directory or a new module is added to the project? Do we need to change the build script?
*   What happens to the build script if we add new source files?
*   What happens if we need a new product, like a new library or a new executable file?

A good build system should handle all or most of the situations covered above. Let's present our first Makefile. This file is going to build the above project and generate its products. All the files written for build systems, in this section and the following sections, can be used to build this particular project and nothing more than that.

The following code box shows the content of the simplest Makefile that we can write for the above project:

```cpp
build:
    mkdir -p out
    gcc -c calc/add.c -o out/add.o
    gcc -c calc/multiply.c -o out/multiply.o
    gcc -c calc/subtract.c -o out/subtract.o
    ar rcs out/libcalc.a out/add.o out/multiply.o out/subtract.o
    gcc -c -Icalc exec/main.c -o out/main.o
    gcc -Lout -lcalc out/main.o -o out/ex23_1.out
clean:
    rm -rfv out
```

Code Box 23-1 [Makefile-very-simple]: A very simple Makefile written for the target project

The preceding Makefile contains two targets: `build` and `clean`. Targets have a set of commands, which should be executed when the target is summoned. This set of commands is called the *recipe* of the target.

In order to run the commands in a Makefile, we need to use the `make` command. You need to tell the `make` command which target to run, but if you leave it empty, make always executes the first target.

To build the preceding project using the Makefile, it is enough to copy the lines from *Code Box 23-1* to a file named `Makefile` and put it in the root of the project. The content of the project's directory should be similar to what we see in the following shell box:

```cpp
$ tree ex23_1
ex23_1/
├── Makefile
├── calc
│   ├── add.c
│   ├── calc.h
│   ├── multiply.c
│   └── subtract.c
└── exec
    └── main.c
2 directories, 6 files 
$
```

Shell Box 23-3: The files and directories found in the target project after adding the Makefile

Following that, you can just run the make command. The `make` program automatically looks for the `Makefile` file in the current directory and executes its first target. If we wanted to run the `clean` target, we would have to use the `make clean` command. The `clean` target can be used to remove the files produced as part of the build process, and this way, we can start a fresh build from scratch.

The following shell box shows the result of running the `make` command:

```cpp
$ cd ex23_1
$ make
mkdir -p out
gcc -c -Icalc exec/main.c -o out/main.o
gcc -c calc/add.c -o out/add.o
gcc -c calc/multiply.c -o out/multiply.o
gcc -c calc/subtract.c -o out/subtract.o
ar rcs out/libcalc.a out/add.o out/multiply.o out/subtract.o
gcc -Lout -lcalc out/main.o -o out/ex23_1.out
$
```

Shell Box 23-4: Building the target project using the very simple Makefile

You might ask, "What is the difference between a build script (written in a shell script), and the above Makefile?" You'd be right to ask this! The preceding Makefile does not represent the way we usually use Make to build our projects.

In fact, the preceding Makefile is a naive usage of the Make build system, and it doesn't benefit from the features we know that Make has to offer.

In other words, so far, a Makefile has been remarkably similar to a shell script, and we could still just use shell scripting (though, of course, that would involve more effort). Now we get to the point where Makefiles become interesting and really different.

The following Makefile is still simple, but it introduces more of the aspects of the Make build system that we are interested in:

```cpp
CC = gcc
build: prereq out/main.o out/libcalc.a
    ${CC} -Lout -lcalc out/main.o -o out/ex23_1.out
prereq:
    mkdir -p out
out/libcalc.a: out/add.o out/multiply.o out/subtract.o
    ar rcs out/libcalc.a out/add.o out/multiply.o out/subtract.o
out/main.o: exec/main.c calc/calc.h
    ${CC} -c -Icalc exec/main.c -o out/main.o
out/add.o: calc/add.c calc/calc.h
    ${CC} -c calc/add.c -o out/add.o
out/subtract.o: calc/subtract.c calc/calc.h
    ${CC} -c calc/subtract.c -o out/subtract.o
out/multiply.o: calc/multiply.c calc/calc.h
    ${CC} -c calc/multiply.c -o out/multiply.o
clean: out
    rm -rf out
```

Code Box 23-2 [Makefile-simple]: A new, but still simple, Makefile written for the target project

As you can see, we can declare a variable in a Makefile and use it in various places, just as we have declared CC in the preceding code box. Variables, together with conditions in a Makefile, allow us to write flexible build instructions with less effort than it takes to write a shell script that would achieve the same flexibility.

Another cool feature of Makefiles is the ability to include other Makefiles. This way, you can benefit from existing Makefiles that you have written in your previous projects.

As you can see in the preceding Makefile, each Makefile can have several targets. Targets start at the beginning of a line and end with a colon, ":". One tab character *must* be used to indent all the instructions within a target (the recipe) in order to make them recognizable by the `make` program. Here is the cool thing about targets: they can be dependent on the other targets.

For example, in the preceding Makefile, the `build` target depends on the `prereq`, `out /main.o`, and `out/libcalc.a` targets. Then, whenever the `build` target is invoked, first, its depending targets will be checked, and if they are not yet produced, then those targets will be invoked first. Now, if you pay more attention to the targets in the preceding Makefile, you should be able to see the flow of execution between targets.

This is definitely something that we miss in a shell script; a lot of control flow mechanisms (loops, conditions, and so on) would be needed to make a shell script work like this. Makefiles are less verbose and more declarative, and that is why we use them. We want to only declare what needs to be built, and we do not need to know about the path it takes to get built. While this is not totally achieved by using Make, it is a start to having a fully featured build system.

Another feature of the targets in a Makefile is that if they are referring to a file or a directory on the disk, such as `out/multiply.o`, the `make` program checks for recent modifications to that file or directory, and if there is no modification since the last build, it skips that target. This is also true for the dependency of `out/multiply.o`, which is `calc/multiply.c`. If the source file, `calc/multiply.c`, has not been changed recently and it has been compiled before, it doesn't make sense to compile it again. This is again a feature that you cannot simply obtain by writing shell scripts.

By having this feature, you only compile the source files that have been modified from the last build, and this reduces a huge amount of compilation for sources that have not been changed since the last build. Of course, this feature will work after having the whole project compiled at least once. After that, only modified sources will trigger a compilation or linkage.

Another crucial point in the preceding Makefile is the `calc/calc.h` target. As you can see, there are multiple targets, mostly source files, that are dependent on the header file, `calc/calc.h`. Therefore, based on the functionality we explained before, a simple modification to the header file can trigger multiple compilations for the source files depending on that header file.

This is exactly why we try to include only the required header files in a source file, and use forward declarations wherever possible instead of inclusion. Forward declarations are not usually made in source files because there, we often demand access to the actual definition of a structure or a function, but it can be easily done in header files.

Having a lot of dependencies between header files usually leads to build disasters. Even a small modification to a header file included by many other header files, and eventually included by many source files, can trigger building the whole project or something on that scale. This will effectively reduce the quality of development as well as lead to a developer having to wait for minutes between builds.

The preceding Makefile is still too verbose. We have to change the targets whenever we add a new source file. We expect to change the Makefile upon adding a new source file, but not by adding a new target and changing the overall structure of a Makefile. This effectively prevents us from reusing the same Makefile in another project similar to the current one.

More than that, many targets follow the same pattern, and we can benefit from the *pattern matching* feature available in Make to reduce the number of targets and write less code in a Makefile. This is another super feature of Make whose effect you cannot easily achieve by writing shell scripts.

The following Makefile will be our last one for this project, but still is not the best Makefile that a Make professional can write:

```cpp
BUILD_DIR = out
OBJ = ${BUILD_DIR}/calc/add.o \
                ${BUILD_DIR}/calc/subtract.o \
                ${BUILD_DIR}/calc/multiply.o \
                ${BUILD_DIR}/exec/main.o
CC = gcc
HEADER_DIRS = -Icalc
LIBCALCNAME = calc
LIBCALC = ${BUILD_DIR}/lib${LIBCALCNAME}.a
EXEC = ${BUILD_DIR}/ex23_1.out
build: prereq ${BUILD_DIR}/exec/main.o ${LIBCALC}
    ${CC} -L${BUILD_DIR} -l${LIBCALCNAME} ${BUILD_DIR}/exec/main.o -o ${EXEC}
prereq:
    mkdir -p ${BUILD_DIR}
    mkdir -p ${BUILD_DIR}/calc
    mkdir -p ${BUILD_DIR}/exec
${LIBCALC}: ${OBJ}
    ar rcs ${LIBCALC} ${OBJ}
${BUILD_DIR}/calc/%.o: calc/%.c
    ${CC} -c ${HEADER_DIRS} $< -o $@
${BUILD_DIR}/exec/%.o: exec/%.c
    ${CC} -c ${HEADER_DIRS} $< -o $@
clean: ${BUILD_DIR}
    rm -rf ${BUILD_DIR}
```

Code Box 23-3 [Makefile-by-pattern]: A new Makefile written for the target project that uses pattern matching

The preceding Makefile uses pattern matching in its targets. The variable `OBJ` keeps a list of the expected relocatable object files, and it is used in all other places when a list of object files is needed.

This is not a book on how Make's pattern matching works, but you can see that there are a bunch of wildcard characters, such as `%`, `$<`, and `$@`, that are used in the patterns.

Running the preceding Makefile will produce the same results as the other Makefiles, but we can benefit from the various nice features that Make offers, and eventually have a reusable and maintainable Make script.

The following shell box shows how to run the preceding Makefile and what the output is:

```cpp
$ make
mkdir -p out
mkdir -p out/calc
mkdir -p out/exec
gcc -c -Icalc exec/main.c -o out/exec/main.o
gcc -c -Icalc calc/add.c -o out/calc/add.o
gcc -c -Icalc calc/subtract.c -o out/calc/subtract.o
gcc -c -Icalc calc/multiply.c -o out/calc/multiply.o
ar rcs out/libcalc.a out/calc/add.o out/calc/subtract.o out/calc/multiply.o out/exec/main.o
gcc -Lout -lcalc out/exec/main.o -o out/ex23_1.out
$
```

Shell Box 23-5: Building the target project using the final Makefile

In the following sections, we will be talking about CMake, a great tool for generating true Makefiles. In fact, a while after Make became popular, a new generation of build tools emerged, *build script generators*, which could generate Makefiles or scripts from other build systems based on a given description. CMake is one of them, and it is probably the most popular one.

**Note**:

Here is the main link to read more about GNU Make, which is the implementation of Make made [for the GNU project: https://www.gnu.org/software/make/manual](https://www.gnu.org/software/make/manual/html_node/index.html)/html_node/index.html.

# CMake – not a build system!

CMake is a build script generator and acts as a generator for other build systems such as Make and Ninja. It is a tedious and complex job to write effective and cross-platform Makefiles. CMake or similar tools, like *Autotools*, are developed to deliver finely tuned cross-platform build scripts such as Makefiles or Ninja build files. Note that Ninja is another build system and will be introduced in the next section.

**Note**:

You can read more [about Autotools here: https://www.gnu.org/software/automake/manual/html_node/Aut](https://www.gnu.org/software/automake/manual/html_node/Autotools-Introduction.html)otools-Introduction.html.

Dependency management is also important, which is not delivered through Makefiles. These generator tools can also check for installed dependencies and won't generate the build scripts if a required dependency is missing from the system. Checking the compilers and their versions, and finding their locations, their supported features, and so on is all part of what these tools do before generating a build script.

Like Make, which looks for a file named `Makefile`, CMake looks for a file named `CMakeLists.txt`. Wherever you find this file in a project, it means that CMake can be used to generate proper Makefiles. Fortunately, and unlike Make, CMake supports nested modules. In other words, you can have multiple `CMakeLists.txt` in other directories as part of your project and all of them can be found and proper Makefiles would be generated for all of them, just by running CMake in the root of your project.

Let's continue this section by adding CMake support to our example project. For this purpose, we add three `CMakeLists.txt` files. Next, you can see the hierarchy of the project after adding these files:

```cpp
$ tree ex23_1
ex23_1/
├── CMakeLists.txt
├── calc
│   ├── CMakeLists.txt
│   ├── add.c
│   ├── calc.h
│   ├── multiply.c
│   └── subtract.c
└── exec
    ├── CMakeLists.txt
    └── main.c
2 directories, 8 files
$
```

Shell Box 23-6: The project hierarchy after introducing three CMakeLists.txt files

As you can see, we have three `CMakeLists.txt` files: one in the root directory, one in the `calc` directory, and the other one in the `exec` directory. The following code box shows the content of the `CMakeLists.txt` file found in the root directory. As you can see, it adds subdirectories of `calc` and `exec`.

These subdirectories must have a `CMakeLists.txt` file inside and, in fact, they do, according to our setup:

```cpp
cmake_minimum_required(VERSION 3.8)
include_directories(calc)
add_subdirectory(calc)
add_subdirectory(exec)
```

Code Box 23-4 [CMakeLists.txt]: The CMakeLists.txt file found in the root directory of the project

The preceding CMake file adds the `calc` directory to the `include` directories that will be used by the C compiler when compiling the source files. Like we said before, it also adds two subdirectories: `calc` and `exec`. These directories have their own `CMakeLists.txt` files that explain how to compile their content. The following is the `CMakeLists.txt` file found in the `calc` directory:

```cpp
add_library(calc STATIC
  add.c
  subtract.c
  multiply.c
)
```

Code Box 23-5 [calc/CMakeLists.txt]: The CMakeLists.txt file found in the calc directory

As you can see, it is just a simple *target declaration* for the `calc` target, meaning that we need to have a static library named `calc` (actually `libcalc.a` after build) that should contain the corresponding relocatable object files for the source files, `add.c`, `subtract.c`, and `multiply.c`. Note that CMake targets usually represent the final products of a code base. Therefore, specifically for the `calc` module, we have only one product, which is a static library.

As you can see, nothing else is specified for the `calc` target. For instance, we didn't specify the extension of the static library or the filename of the library (even though we could). All other configurations required to build this module are either inherited from the parent `CMakeLists.txt` file or have been obtained from the default configuration of CMake itself.

For example, we know that the extension for shared object files is different on Linux and macOS. Therefore, if the target is a shared library, there is no need to specify the extension as part of the target declaration. CMake is able to handle this very platform-specific difference, and the final shared object file will have the correct extension based on the platform that it is being built on.

The following `CMakeLists.txt` file is the one found in the `exec` directory:

```cpp
add_executable(ex23_1.out
  main.c
)
target_link_libraries(ex23_1.out
  calc
)
```

Code Box 23-6 [exec/CMakeLists.txt]: The CMakeLists.txt file found in the exec directory

As you can see, the target declared in the preceding `CMakeLists.txt` is an executable, and it should be linked to the `calc` target that is already declared in another `CMakeLists.txt` file.

This really gives you the power to create libraries in one corner of your project and use them in another corner just by writing some directives.

Now it's time to show you how to generate a Makefile based on the `CMakeLists.txt` file found in the root directory. Note that we do this in a separate directory named `build` in order to have the resulting relocatable and final object files kept separated from the actual sources.

If you're using a **source control management** (**SCM**) system like *git*, you can ignore the `build` directory because it should be generated on each platform separately. The only files that matter are the `CMakeLists.txt` files, which are always kept in a source control repository.

The following shell box demonstrates how to generate build scripts (in this case, a Makefile) for the `CMakeLists.txt` file found in the root directory:

```cpp
$ cd ex23_1
$ mkdir -p build
$ cd build
$ rm -rfv *
...
$ cmake ..
-- The C compiler identification is GNU 7.4.0
-- The CXX compiler identification is GNU 7.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: .../extreme_c/ch23/ex23_1/build
$
```

Shell Box 23-7: Generating a Makefile based on the CMakeLists.txt file found in the root directory

As you can see from the output, the CMake command has been able to detect the working compilers, their ABI info (for more on ABI, refer to *Chapter 3*, *Object Files*), their features, and so on, and finally it has generated a Makefile in the `build` directory.

**Note**:

In *Shell Box 23-7*, we assumed that we could have had the `build` directory in place; therefore, we removed all of its content first.

You can see the content of the `build` directory and the generated Makefile:

```cpp
$ ls
CMakeCache.txt  CMakeFiles  Makefile  calc  cmake_install.cmake  exec
$
```

Shell Box 23-8: Generated Makefile in the build directory

Now that you've got a Makefile in your `build` directory, you're free to run the make command. It will take care of the compilation and display its progress nicely for you.

Note that you should be in the `build` directory before running the `make` command:

```cpp
$ make
Scanning dependencies of target calc
[ 16%] Building C object calc/CMakeFiles/calc.dir/add.c.o
[ 33%] Building C object calc/CMakeFiles/calc.dir/subtract.c.o
[ 50%] Building C object calc/CMakeFiles/calc.dir/multiply.c.o
[ 66%] Linking C static library libcalc.a
[ 66%] Built target calc
Scanning dependencies of target ex23_1.out
[ 83%] Building C object exec/CMakeFiles/ex23_1.out.dir/main.c.o
[100%] Linking C executable ex23_1.out
[100%] Built target ex23_1.out
$
```

Shell Box 23-9: Executing the generated Makefile

Currently, many big projects use CMake, and you can build their sources by using more or less the same commands that we've shown in the previous shell boxes. *Vim* is one such project. Even CMake itself is built using CMake after having a minimum CMake system built by Autotools! CMake now has lots of versions and features and it would take a whole book to discuss them in extensive detail.

**Note**:

The following link is the official documentation of the latest version of CMake and it can help you to get an idea of how it [works and what features it has: https://cmake.](https://cmake.org/cmake/help/latest/index.html)org/cmake/help/latest/index.html.

As a final note in this section, CMake can create build script files for Microsoft Visual Studio, Apple's Xcode, and other development environments.

In the following section, we will be discussing the Ninja build system, a fast alternative to Make that has been gaining momentum recently. We also explain how CMake can be used to generate Ninja build script files instead of Makefiles.

# Ninja

Ninja is an alternative to Make. I hesitate to call it a replacement, but it is a faster alternative. It achieves its high performance by removing some of the features that Make offers, such as string manipulation, loops, and pattern matching.

Ninja has less overhead by removing these features, and because of that, it is not wise to write Ninja build scripts from scratch.

Writing Ninja scripts can be compared to writing shell scripts, the downsides of which we explained in the previous section. That's why it is recommended to use it together with a build script generator tool like CMake.

In this section, we show how Ninja can be used when Ninja build scripts are generated by CMake. Therefore, in this section, we won't go through the syntax of Ninja, as we did for Makefiles. That's because we are not going to write them ourselves; instead, we are going to ask CMake to generate them for us.

**Note**:

For more information on Ninja syntax, please follow this link: https://ninja-build.org/manual.html#_writing_your_own_ninja_files.

As we explained before, it is best to use a build script generator to produce Ninja build script files. In the following shell box, you can see how to use CMake to generate a Ninja build script, `build.ninja`, instead of a Makefile for our target project:

```cpp
$ cd ex23_1
$ mkdir -p build
$ cd build
$ rm -rfv *
...
$ cmake -GNinja ..
-- The C compiler identification is GNU 7.4.0
-- The CXX compiler identification is GNU 7.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done
-- Generating done
-- Build files have been written to: .../extreme_c/ch23/ex23_1/build
$
```

Shell Box 23-10: Generating build.ninja based on CMakeLists.txt found in the root directory

As you can see, we have passed the option `-GNinja` to let CMake know that we are demanding Ninja build script files instead of Makefiles. CMake generates the `build.ninja` file and you can find it in the `build` directory as follows:

```cpp
$ ls
CMakeCache.txt  CMakeFiles  build.ninja  calc  cmake_install.cmake  exec  rules.ninja
$
```

Shell Box 23-11: Generated build.ninja in the build directory

To compile the project, it is enough to run the `ninja` command as follows. Note that just as the `make` program looks for the `Makefile` in the current directory, the `ninja` program looks for `build.ninja` in the current directory:

```cpp
$ ninja
[6/6] Linking C executable exec/ex23_1.out
$
```

Shell Box 23-12: Executing generated build.ninja

In the following section, we are going to talk about *Bazel*, another build system that can be used for building C and C++ projects.

# Bazel

Bazel is a build system developed at Google to address the internal need to have a fast and scalable build system that can build any project no matter what the programming language is. Bazel supports building C, C++, Java, Go, and Objective-C projects. More than that, it can be used to build Android and iOS projects.

Bazel became open source around 2015\. It is a build system, so it can be compared with Make and Ninja, but not CMake. Almost all of Google's open source projects use Bazel for their builds. For example, we can name *Bazel* itself, *gRPC*, *Angular*, *Kubernetes*, and *TensorFlow*.

Bazel is written in Java. It is famous for parallel and scalable builds, and it really makes a difference in big projects. Parallel builds are also available in Make and Ninja, both by passing the `-j` option (Ninja is parallel by default, however).

**Note**:

The official docume[ntation of Bazel can be found here: https://docs.bazel.buil](https://docs.bazel.build/versions/master/bazel-overview.html)d/versions/master/bazel-overview.html.

The way to use Bazel is similar to what we did for Make and Ninja. Bazel requires two kinds of files to be present in a project: `WORKSPACE` and `BUILD` files. The `WORKSPACE` file should be in the root directory, and the `BUILD` files should be put into the modules that should be built as part of the same workspace (or project). This is more or less similar to the case with CMake, where we had three `CMakeLists.txt` files distributed in the project, but note that, here, Bazel itself is the build system and we are not going to generate any build script for another build system.

If we want to add the Bazel support to our project, we should obtain the following hierarchy in the project:

```cpp
$ tree ex23_1
ex23_1/
├── WORKSPACE
├── calc
│   ├── BUILD
│   ├── add.c
│   ├── calc.h
│   ├── multiply.c
│   └── subtract.c
└── exec
    ├── BUILD
    └── main.c
2 directories, 8 files
$
```

Shell Box 23-13: The project hierarchy after introducing Bazel files

The content of the `WORKSPACE` file would be empty in our example. It is usually used to indicate the root of the code base. Note that you need to refer to the documentation to see how these files, `WORKSPACE` and `BUILD`, should be propagated throughout the code base if you have even more nested and deeper modules.

The content of the `BUILD` file indicates the targets that should be built in that directory (or module). The following code box shows the `BUILD` file for the `calc` module:

```cpp
c_library(
  name = "calc",
  srcs = ["add.c", "subtract.c", "multiply.c"],
  hdrs = ["calc.h"],
  linkstatic = True,
  visibility = ["//exec:__pkg__"]
)
```

Code Box 23-7 [calc/BUILD]: The BUILD file found in the calc directory

As you see, a new target, `calc`, is declared. It is a static library and contains the three source files found in the directory. The library is also visible to the targets residing in the `exec` directory.

Let's look at the `BUILD` file in the exec directory:

```cpp
cc_binary(
  name = "ex23_1.out",
  srcs = ["main.c"],
  deps = [
    "//calc:calc"
  ],
  copts = ["-Icalc"]
)
```

Code Box 23-8 [exec/BUILD]: The BUILD file found in the exec directory

With the preceding files in their places, we can now run Bazel and build the project. You need to go to the project's root directory. Note that there is no need to have a build directory as we did for CMake:

```cpp
$ cd ex23_1
$ bazel build //...
INFO: Analyzed 2 targets (14 packages loaded, 71 targets configured).
INFO: Found 2 targets...
INFO: Elapsed time: 1.067s, Critical Path: 0.15s
INFO: 6 processes: 6 linux-sandbox.
INFO: Build completed successfully, 11 total actions
$
```

Shell Box 23-14: Building the example project using Bazel

Now, if you look at the `bazel-bin` directory found in the root directory, you should be able to find the products:

```cpp
$ tree bazel-bin
bazel-bin
├── calc
│   ├── _objs
│   │   └── calc
│   │       ├── add.pic.d
│   │       ├── add.pic.o
│   │       ├── multiply.pic.d
│   │       ├── multiply.pic.o
│   │       ├── subtract.pic.d
│   │       └── subtract.pic.o
│   ├── libcalc.a
│   └── libcalc.a-2.params
└── exec
    ├── _objs
    │   └── ex23_1.out
    │       ├── main.pic.d
    │       └── main.pic.o
    ├── ex23_1.out
    ├── ex23_1.out-2.params
    ├── ex23_1.out.runfiles
    │   ├── MANIFEST
    │   └── __main__
    │       └── exec
    │           └── ex23_1.out -> .../bin/exec/ex23_1.out
    └── ex23_1.out.runfiles_manifest
9 directories, 15 files
$
```

Shell Box 23-15: The content of bazel-bin after running the build

As you can see in the preceding list, the project is built successfully, and the products have been located.

In the next section, we are going to close our discussion in this chapter and compare various build systems that exist for C and C++ projects.

# Comparing build systems

In this chapter, we tried to introduce three of the most well-known and widely used build systems. We also introduced CMake as a build script generator. You should know that there are other build systems that can be used to build C and C++ projects.

Note that your choice of build system should be considered as a long-term commitment; if you start a project with a specific build system, it would take significant effort to change it to another one.

Build systems can be compared based on various properties. Dependency management, being able to handle a complex hierarchy of nested projects, build speed, scalability, integration with existing services, flexibility to add a new logic, and so on can all be used to make a fair comparison. I'm not going to finish this book with a comparison of build systems because it is a tedious job to do, and, more than that, there are already some great online articles covering the topic.

A nice Wiki page on Bitbucket that does a pros/cons comparison on available build systems, together with build script [generator systems can be found here: https://bitbucket.o](https://bitbucket.org/scons/scons/wiki/SconsVsOtherBuildTools)rg/scons/scons/wiki/SconsVsOtherBuildTools.

Note that the result of a comparison can be different for anyone. You should choose a build system based on your project's requirements and the resources available to you. The following links lead to supplementary resources that can [be used for further study and comparison:](https://www.reddit.com/r/cpp/comments/8zm66h/an_overview_of_build_systems_mostly_for_c_projects/)

[https://www.reddit.com/r/cpp/comments/8zm66h/an_overv](https://www.reddit.com/r/cpp/comments/8zm66h/an_overview_of_build_systems_mostly_for_c_projects/)i[ew_of_build_systems_mostly_for_c_projects/](https://github.com/LoopPerfect/buckaroo/wiki/Build-Systems-Comparison)

[https://github.com/LoopPer](https://github.com/LoopPerfect/buckaroo/wiki/Build-Systems-Comparison)f[ect/buckaroo/wiki/Build-Systems-Comparison](https://medium.com/@julienjorge/an-overview-of-build-systems-mostly-for-c-projects-ac9931494444)

[https://medium.com/@julienjorge/an-overview-of-build](https://medium.com/@julienjorge/an-overview-of-build-systems-mostly-for-c-projects-ac9931494444)-systems-mostly-for-c-projects-ac9931494444

# Summary

In this chapter, we discussed the common build tools available for building a C or C++ project. As part of this chapter:

*   We discussed the need for a build system.
*   We introduced Make, one of the oldest build systems available for C and C++ projects.
*   We introduced Autotools and CMake, two famous build script generators.
*   We showed how CMake can be used to generate the required Makefiles.
*   We discussed Ninja and we showed how CMake can be used to generate Ninja build scripts.
*   We demonstrated how Bazel can be used to build a C project.
*   Finally, we provided some links to a number of online discussions regarding the comparison of various build systems.

# Epilogue

And the final words ...

If you are reading this, it means that our journey has come to an end! We went through several topics and concepts as part of this book, and I hope that the journey has made you a better C programmer. Of course, it cannot give you the experience; you must obtain that by working on various projects. The methods and tips we discussed in this book will ramp up your level of expertise, and this will enable you to work on more serious projects. Now you know more about software systems, from a broader point of view, and possess a top-notch knowledge about the internal workings.

Though this book was heavier and lengthier than your usual read, it still could not cover all the topics found within C, C++, and system programming. Therefore, a weight remains on my shoulders; the journey is not yet done! I would like to continue to work on more Extreme topics, maybe more specific areas, such as Asynchronous I/O, Advanced Data Structures, Socket Programming, Distributed Systems, Kernel Development, and Functional Programming, in time.

Hope to see you again on the next journey!

Kamran