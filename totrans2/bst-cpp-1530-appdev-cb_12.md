# Chapter 12. Scratching the Tip of the Iceberg

In this chapter we will cover:

*   Working with graphs
*   Visualizing graphs
*   Using a true random number generator
*   Using portable math functions
*   Writing test cases
*   Combining multiple test cases in one test module
*   Manipulating images

# Introduction

Boost is a huge collection of libraries. Some of those libraries are small and meant for everyday use and others require a separate book to describe all of their features. This chapter is devoted to some of those big libraries and to give you some basics to start with.

The first two recipes will explain the usage of `Boost.Graph` . It is a big library with an insane number of algorithms. We'll see some basics and probably the most important part of it visualization of graphs.

We'll also see a very useful recipe for generating true random numbers. This is a very important requirement for writing secure cryptography systems.

Some C++ standard libraries lack math functions. We'll see how that can be fixed using Boost. But the format of this book leaves no space to describe all of the functions.

Writing test cases is described in the *Writing test cases* and *Combining multiple test cases in one test module* recipes. This is important for any production-quality system.

The last recipe is about a library that helped me in many courses during my university days. Images can be created and modified using it. I personally used it to visualize different algorithms, hide data in images, sign images, and generate textures.

Unfortunately, even this chapter cannot tell you about all of the Boost libraries. Maybe someday I'll write another book... and then a few more.

# Working with graphs

Some tasks require a graphical representation of data. `Boost.Graph` is a library that was designed to provide a flexible way of constructing and representing graphs in memory. It also contains a lot of algorithms to work with graphs, such as topological sort, breadth first search, depth first search, and Dijkstra shortest paths.

Well, let's perform some basic tasks with `Boost.Graph`!

## Getting ready

Only basic knowledge of C++ and templates is required for this recipe.

## How to do it...

In this recipe, we'll describe a graph type, create a graph of that type, add some vertexes and edges to the graph, and search for a specific vertex. That should be enough to start using `Boost.Graph`.

1.  We start with describing the graph type:

    [PRE0]

2.  Now we construct it:

    [PRE1]

3.  Let's use a non portable trick that speeds up graph construction:

    [PRE2]

4.  Now we are ready to add vertexes to the graph:

    [PRE3]

5.  It is time to connect vertexes with edges:

    [PRE4]

6.  We make a function that searches for a vertex:

    [PRE5]

7.  Now we will write code that gets iterators to all vertexes:

    [PRE6]

8.  It's time to run a search for the required vertex:

    [PRE7]

## How it works...

In step 1, we are describing what our graph must look like and upon what types it must be based. `boost::adjacency_list` is a class that represents graphs as a two-dimensional structure, where the first dimension contains vertexes and the second dimension contains edges for that vertex. `boost::adjacency_list` must be the default choice for representing a graph; it suits most cases.

The first template parameter, `boost::adjacency_list`, describes the structure used to represent the edge list for each of the vertexes; the second one describes a structure to store vertexes. We can choose different STL containers for those structures using specific selectors, as listed in the following table:

| Selector | STL container |
| --- | --- |
| `boost::vecS` | `std::vector` |
| `boost::listS` | `std::list` |
| `boost::slistS` | `std::slist` |
| `boost::setS` | `std::set` |
| `boost::multisetS` | `std::multiset` |
| `boost::hash_setS` | `std::hash_set` |

The third template parameter is used to make an undirected, directed, or bidirectional graph. Use the `boost::undirectedS`, `boost::directedS`, and `boost::bidirectionalS` selectors respectively.

The fifth template parameter describes the datatype that will be used as the vertex. In our example, we chose `std::string`. We can also support a datatype for edges and provide it as a template parameter.

Steps 2 and 3 are trivial, but at step 4 you will see a non portable way to speed up graph construction. In our example, we use `std::vector` as a container for storing vertexes, so we can force it to reserve memory for the required amount of vertexes. This leads to less memory allocations/deallocations and copy operations during insertion of vertexes into the graph. This step is non-portable because it is highly dependent on the current implementation of `boost::adjacency_list` and on the chosen container type for storing vertexes.

At step 4, we see how vertexes can be added to the graph. Note how `boost::graph_traits<graph_type>` has been used. The `boost::graph_traits` class is used to get types that are specific for a graph type. We'll see its usage and the description of some graph-specific types later in this chapter. Step 5 shows what we need do to connect vertexes with edges.

### Note

If we had provided a datatype for the edges, adding an edge would look as follows:

`boost::add_edge(ansic, guru, edge_t(initialization_parameters), graph)`

Note that at step 6 the graph type is a `template` parameter. This is recommended to achieve better code reusability and make this function work with other graph types.

At step 7, we see how to iterate over all of the vertexes of the graph. The type of vertex iterator is received from `boost::graph_traits`. The function `boost::tie` is a part of `Boost.Tuple` and is used for getting values from tuples to the variables. So calling `boost::tie(it, end) = boost::vertices(g)` will put the `begin` iterator into the `it` variable and the `end` iterator into the `end` variable.

It may come as a surprise to you, but dereferencing a vertex iterator does not return vertex data. Instead, it returns the vertex descriptor `desc`, which can be used in `boost::get(boost::vertex_bundle, g)[desc]` to get vertex data, just as we have done in step 8\. The vertex descriptor type is used in many of the `Boost.Graph` functions; we saw its use in the edge construction function in step 5.

### Note

As already mentioned, the `Boost.Graph` library contains the implementation of many algorithms. You will find many search policies implemented, but we won't discuss them in this book. We will limit this recipe to only the basics of the graph library.

## There's more...

The `Boost.Graph` library is not a part of C++11 and it won't be a part of C++1y. The current implementation does not support C++11 features. If we are using vertexes that are heavy to copy, we may gain speed using the following trick:

[PRE8]

It avoids copy constructions of `boost::add_vertex(vertex_data, graph)` and uses the default construction with `move` assignment instead.

The efficiency of `Boost.Graph` depends on multiple factors, such as the underlying containers types, graph representation, edge, and vertex datatypes.

## See also

*   Reading the *Visualizing graphs* recipe can help you work more easily with graphs. You may also consider reading its official documentation at the following link:

    [http://www.boost.org/doc/libs/1_53_0/libs/graph/doc/table_of_contents.html](http://www.boost.org/doc/libs/1_53_0/libs/graph/doc/table_of_contents.html)

# Visualizing graphs

Making programs that manipulate graphs was never easy because of issues with visualization. When we work with STL containers such as `std::map` and `std::vector`, we can always print the container's contents and see what is going on inside. But when we work with complex graphs, it is hard to visualize the content in a clear way: too many vertexes and too many edges.

In this recipe, we'll take a look at the visualization of `Boost.Graph` using the **Graphviz** tool.

## Getting ready

To visualize graphs, you will need a Graphviz visualization tool. Knowledge of the preceding recipe is also required.

## How to do it...

Visualization is done in two phases. In the first phase, we make our program output the graph's description in a text format; in the second phase, we import the output from the first step to some visualization tool. The numbered steps in this recipe are all about the first phase.

1.  Let's write the `std::ostream` operator for `graph_type` as done in the preceding recipe:

    [PRE9]

2.  The `detail::vertex_writer` structure, used in the preceding step, must be defined as follows:

    [PRE10]

That's all. Now, if we visualize the graph from the previous recipe using the `std::cout << graph;` command, the output can be used to create graphical pictures using the `dot` command-line utility:

[PRE11]

The output of the preceding command is depicted in the following figure:

![How to do it...](img/4880OS_12_02.jpg)

We can also use the **Gvedit** or **XDot** programs for visualization if the command line frightens you.

## How it works...

The `Boost.Graph` library contains function to output graphs in Graphviz (DOT) format. If we write `boost::write_graphviz(out, g)` with two parameters in step 1, the function will output a graph picture with vertexes numbered from `0`. That's not very useful, so we provide an instance of the `vertex_writer` class that outputs vertex names.

As we can see in step 2, the format of output must be DOT, which is understood by the Graphviz tool. You may need to read the Graphviz documentation for more info about the DOT format.

If you wish to add some data to the edges during visualization, we need to provide an instance of the edge visualizer as a fourth parameter to `boost::write_graphviz`.

## There's more...

C++11 does not contain `Boost.Graph` or the tools for graph visualization. But you do not need to worry—there are a lot of other graph formats and visualization tools and `Boost.Graph` can work with plenty of them.

## See also

*   The *Working with graphs* recipe contains information about the construction of `Boost.Graphs`
*   You will find a lot of information about the DOT format and Graphviz at [http://www.graphviz.org/](http://www.graphviz.org/)
*   Boost's official documentation for the `Boost.Graph` library contains multiple examples and useful information, and can be found at [http://www.boost.org/doc/libs/1_53_0/libs/graph/doc/table_of_](http://www.boost.org/doc/libs/1_53_0/libs/graph/doc/table_of_)[contents.html](http://contents.html)

# Using a true random number generator

I know of many examples of commercial products that use incorrect methods for getting random numbers. It's a shame that some companies still use `rand()` in cryptography and banking software.

Let's see how to get a fully random uniform distribution using `Boost.Random` that is suitable for banking software.

## Getting ready

Basic knowledge of C++ is required for this recipe. Knowledge of different types of distributions will also be helpful. The code in this recipe requires linking against the `boost_random` library.

## How to do it...

To create a true random number, we need some help from the operating system or processor. This is how it can be done using Boost:

1.  We'll need to include the following headers:

    [PRE12]

2.  Advanced random number providers have different names under different platforms:

    [PRE13]

3.  Now we are ready to initialize the generator with `Boost.Random`:

    [PRE14]

4.  Let's get a uniform distribution that returns a value between 1000 and 65535:

    [PRE15]

That's it. Now we can get true random numbers using the `random(device)` call.

## How it works...

Why does the `rand()` function not suit banking? Because it generates pseudo-random numbers, which means that the hacker could predict the next generated number. This is an issue with all pseudo-random number algorithms. Some algorithms are easier to predict and some harder, but it's still possible.

That's why we are using `boost::random_device` in this example (see step 3). That device gathers information about random events from all around the operating system to construct an unpredictable hardware-generated number. The examples of such events are delays between pressed keys, delays between some of the hardware interruptions, and the internal CPU random number generator.

Operating systems may have more than one such type of random number generators. In our example for POSIX systems, we used `/dev/urandom` instead of the more secure `/dev/random` because the latter remains in a blocked state until enough random events have been captured by the OS. Waiting for entropy could take seconds, which is usually unsuitable for applications. Use `/dev/random` to create long-lifetime `GPG/SSL/SSH` keys.

Now that we are done with generators, it's time to move to step 4 and talk about distribution classes. If the generator just generates numbers (usually uniformly distributed), the distribution class maps one distribution to another. In step 4, we made a uniform distribution that returns a random number of unsigned short type. The parameter `1000` means that distribution must return numbers greater or equal to `1000`. We can also provide the maximum number as a second parameter, which is by default equal to the maximum value storable in the return type.

## There's more...

`Boost.Random` has a huge number of true/pseudo random generators and distributions for different needs. Avoid copying distributions and generators; this could turn out to be an expensive operation.

C++11 has support for different distribution classes and generators. You will find all of the classes from this example in the `<random>` header in the `std::` namespace. The `Boost.Random` libraries do not use C++11 features, and they are not really required for that library either. Should you use Boost implementation or STL? Boost provides better portability across systems; however, some STL implementations may have assembly-optimized implementations and might provide some useful extensions.

## See also

*   The official documentation contains a full list of generators and distributions with descriptions; it is available at the following link:

    [http://www.boost.org/doc/libs/1_53_0/doc/html](http://www.boost.org/doc/libs/1_53_0/doc/html) [/boost_random.html](http:///boost_random.html)

# Using portable math functions

Some projects require specific trigonometric functions, a library for numerically solving ordinary differential equations, and working with distributions and constants. All of those parts of `Boost.Math` would be hard to fit into even a separate book. A single recipe definitely won't be enough. So let's focus on very basic everyday-use functions to work with float types.

We'll write a portable function that checks an input value for infinity and not-a-number (NaN) values and changes the sign if the value is negative.

## Getting ready

Basic knowledge of C++ is required for this recipe. Those who know C99 standard will find a lot in common in this recipe.

## How to do it...

Perform the following steps to check the input value for infinity and NaN values and change the sign if the value is negative:

1.  We'll need the following headers:

    [PRE16]

2.  Asserting for infinity and NaN can be done like this:

    [PRE17]

3.  Use the following code to change the sign:

    [PRE18]

That's it! Now we can check that `check_float_inputs(std::sqrt(-1.0))` and `check_float_inputs(std::numeric_limits<double>::max() * 2.0)` will cause asserts.

## How it works...

Real types have specific values that cannot be checked using equality operators. For example, if the variable `v` contains NaN, `assert(v!=v)` may or may not pass depending on the compiler.

For such cases, `Boost.Math` provides functions that can reliably check for infinity and NaN values.

Step 3 contains the `boost::math::signbit` function, which requires clarification. This function returns a signed bit, which is 1 when the number is negative and 0 when the number is positive. In other words, it returns `true` if the value is negative.

Looking at step 3 some readers might ask, "Why can't we just multiply by `-1` instead of calling `boost::math::changesign`?". We can. But multiplication may work slower than `boost::math::changesign` and won't work for special values. For example, if your code can work with `nan`, the code in step 3 will be able to change the sign of `-nan` and write `nan` to the variable.

### Note

The `Boost.Math` library maintainers recommend wrapping math functions from this example in round parenthesis to avoid collisions with C macros. It is better to write `(boost::math::isinf)(value)` instead of `boost::math::isinf(value)`.

## There's more...

C99 contains all of the functions described in this recipe. Why do we need them in Boost? Well, some compiler vendors think that programmers do not need them, so you won't find them in one very popular compiler. Another reason is that the `Boost.Math` functions canbe used for classes that behave like numbers.

`Boost.Math` is a very fast, portable, reliable library.

## See also

*   Boost's official documentation contains lots of interesting examples and tutorials that will help you get used to `Boost.Math`; browse to [http://www.boost.org/doc/libs/1_53_0/libs/math/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/math/doc/html/index.html)

# Writing test cases

This recipe and the next one are devoted to auto-testing the `Boost.Test` library, which is used by many Boost libraries. Let's get hands-on with it and write some tests for our own class.

[PRE19]

## Getting ready

Basic knowledge of C++ is required for this recipe. The code of this recipe requires linking against the static version of the `boost_unit_test_framework` library.

## How to do it...

To be honest, there is more than one test library in Boost. We'll take a look at the most functional one.

1.  To use it, we'll need to define the macro and include the following header:

    [PRE20]

2.  Each set of tests must be written in the test case:

    [PRE21]

3.  Checking some function for the `true` result is done as follows:

    [PRE22]

4.  Checking for nonequality is implemented in the following way:

    [PRE23]

5.  Checking for an exception being thrown will look like this:

    [PRE24]

That's it! After compilation and linking, we'll get an executable file that automatically tests `foo` and outputs test results in a human-readable format.

## How it works...

Writing unit tests is easy; you know how the function works and what result it should produce in specific situations. So you just check if the expected result is the same as the function's actual output. That's what we did in step 3\. We know that `f1.is_not_null()` will return `true` and we checked it. At step 4, we know that `f1` is not equal to `f2`, so we checked it too. The call to `f1.throws()` will produce the `std::logic_error` exception and we check that an exception of the expected type is thrown.

At step 2, we are making a test case – a set of checks to validate correct behavior of the `foo` structure. We can have multiple test cases in a single source file. For example, if we add the following code:

[PRE25]

This code will run along with the `test_no_1` test case. The parameter passed to the `BOOST_AUTO_TEST_CASE` macro is just a unique name of the test case that will be shown in case of error.

[PRE26]

There is a small difference between the `BOOST_REQUIRE_*` and `BOOST_CHECK_*` macros. If the `BOOST_REQUIRE_*` macro check fails, the execution of the current test case will stop and `Boost.Test` will run the next test case. However, failing `BOOST_CHECK_*` won't stop the execution of the current test case.

Step 1 requires additional care. Note the `BOOST_TEST_MODULE` macro definition. This macro must be defined before including the `Boost.Test` headers, otherwise linking of the program will fail. More information can be found in the *See also* section of this recipe.

## There's more...

Some readers may wonder, "Why did we write `BOOST_CHECK_NE(f1, f2)` in step 4 instead of `BOOST_CHECK(f1 != f2)`?". The answer is simple: the macro at step 4 provides a more readable and verbose output.

C++11 lacks support for unit testing. However, the `Boost.Test` library can be used to test C++11 code. Remember that the more tests you have, the more reliable code you get!

## See also

*   The *Combining multiple test cases in one test module* recipe contains more information about testing and the `BOOST_TEST_MODULE` macro
*   Refer to Boost's official documentation for a full list of test macros and information about advanced features of `Boost.Test`; it's available at the following link:

    [http://www.boost.org/doc/libs/1_53_0/libs/test/doc/html/index.html](http://www.boost.org/doc/libs/1_53_0/libs/test/doc/html/index.html)

# Combining multiple test cases in one test module

Writing auto tests is good for your project. But managing test cases is hard when the project is large and many developers are working on it. In this recipe, we'll take a look at how to run individual tests and how to combine multiple test cases in a single module.

Let's pretend that two developers are testing the `foo` structure declared in the `foo.hpp` header and we wish to give them separate source files to write a test to. In that way, the developers won't bother each other and can work in parallel. However, the default test run must execute the tests of both developers.

## Getting ready

Basic knowledge of C++ is required for this recipe. This recipe partially reuses code from the previous recipe and it also requires linking against the static version of the `boost_unit_test_framework` library.

## How to do it...

This recipe uses the code from the previous one. This is a very useful recipe for testing large projects; do not underestimate it.

1.  Of all the headers in `main.cpp` from the previous recipe, leave only these two lines:

    [PRE27]

2.  Let's move the tests cases from the previous example into two different source files:

    [PRE28]

That's it! Thus compiling and linking all of the sources and both test cases will work on program execution.

## How it works...

All of the magic is done by the `BOOST_TEST_MODULE` macro. If it is defined before `<boost/test/unit_test.hpp>`, `Boost.Test` thinks that this source file is the main one and all of the helper testing infrastructure must be placed in it. Otherwise, only the test macro will be included from `<boost/test/unit_test.hpp>`.

All of the `BOOST_AUTO_TEST_CASE` tests are run if you link them with the source file that contains the `BOOST_TEST_MODULE` macro. When working on a big project, each developer may enable compilation and linking of only their own sources. That gives independence from other developers and increases the speed of development—no need to compile alien sources and run alien tests while debugging.

## There's more...

The `Boost.Test` library is good because of its ability to run tests selectively. We can choose which tests to run and pass them as command-line arguments. For example, the following command will run only the `test_no_1` test case:

[PRE29]

The following command will run two test cases:

[PRE30]

Unfortunately, C++11 standard does not have built-in testing support and it looks like C++1y won't adopt the classes and methods of `Boost.Test` either.

## See also

*   The *Writing test cases* recipe contains more information about the `Boost.Test` library. Read Boost's official documentation for more information about `Boost.Test`, at [http://www.boost.org/doc/libs/1_53_0/libs/test/doc/html/utf.html](http://www.boost.org/doc/libs/1_53_0/libs/test/doc/html/utf.html).
*   Brave readers can take a look at some of the test cases from the Boost library. Those test cases are allocated in the `libs` subfolder located in the `boost` folder. For example, `Boost.LexicalCast` tests cases are allocated at `boost_1_53_0\libs\conversion\test`.

# Manipulating images

I've left you something really tasty for dessert – Boost's **Generic Image Library** (**GIL**), which allows you to manipulate images and not care much about image formats.

Let's do something simple and interesting with it; let's make a program that negates any picture.

## Getting ready

This recipe requires basic knowledge of C++, templates, and `Boost.Variant`. The example requires linking against the PNG library.

## How to do it...

For simplicity, we'll be working with only PNG images.

1.  Let's start with including the header files:

    [PRE31]

2.  Now we need to define the image types that we wish to work with:

    [PRE32]

3.  Opening an existing PNG image can be implemented like this:

    [PRE33]

4.  We need to apply the operation to the picture as follows:

    [PRE34]

5.  The following code line will help you to write an image:

    [PRE35]

6.  Let's take a look at the modifying operation:

    [PRE36]

7.  The body of `operator()` consists of getting a channel type:

    [PRE37]

8.  It also iterates through pixels:

    [PRE38]

Now let's see the results of our program:

![How to do it...](img/4880OS_12_01.jpg)

The previous picture is the negative of the one that follows:

![How to do it...](img/4880OS_12_03.jpg)

## How it works...

In step 2, we are describing the types of images we wish to work with. Those images are gray images with 8 and 16 bits per pixel and RGB pictures with 8 and 16 bits per pixel.

The `boost::gil::any_image<img_types>` class is a kind of `Boost.Variant` that can hold an image of one of the `img_types` variable. As you may have already guessed, `boost::gil::png_read_image` reads images into image variables.

The `boost::gil::apply_operation` function at step 4 is almost equal to `boost::apply_visitor` from the `Boost.Variant` library. Note the usage of `view(source)`. The `boost::gil::view` function constructs a light wrapper around the image that interprets it as a two-dimensional array of pixels.

Do you remember that for `Boost.Variant` we were deriving visitors from `boost::static_visitor`? When we are using GIL's version of variant, we need to make a `result_type` typedef inside `visitor`. You can see it in step 6.

A little bit of theory: images consist of points called pixels. Single images have pixels of the same type. However, pixels of different images can differ in channel count and color bits for a single channel. A channel represents a primary color. In the case of an RGB image, we'll have a pixel consisting of three channels—red, green, and blue. In the case of a gray image, we'll have a single channel representing gray.

Back to our image. In step 2, we described the types of images we wish to work with. In step 3, one of those image types is read from file and stored in the source variable. In step 4, the `operator()` method of the `negate` visitor is instantiated for all image types.

In step 7, we can see how to get the channel type from the image view.

In step 8, we iterate through pixels and channels and negate them. Negation is done via `max_val - source(x, y)[c]` and the result is written back to the image view.

We write an image back in step 5.

## There's more...

C++11 has no built-in methods for working with images.

The `Boost.GIL` library is fast and efficient. The compilers optimize its code very well and we can even help the optimizer using some of the `Boost.GIL` methods to unroll loops. But this chapter talks about only some of the library basics, so it is time to stop.

## See also

*   More information about `Boost.GIL` can be found at Boost's official documentation; go to [http://www.boost.org/doc/libs/1_53_0/libs/gil/doc/index.html](http://www.boost.org/doc/libs/1_53_0/libs/gil/doc/index.html)
*   See the *Storing multiple chosen types in a variable/container* recipe in [Chapter 1](ch01.html "Chapter 1. Starting to Write Your Application"), *Starting to Write Your Application*, for more information about the `Boost.Variant` library