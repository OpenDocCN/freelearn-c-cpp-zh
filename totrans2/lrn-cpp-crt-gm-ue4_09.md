# Chapter 9. Templates and Commonly Used Containers

In [Chapter 7](part0051_split_000.html#1GKCM1-dd4a3f777fc247568443d5ffb917736d "Chapter 7. Dynamic Memory Allocation"), *Dynamic Memory Allocation*, we spoke about how you will use dynamic memory allocation if you want to create a new array whose size isn't known at compile time. Dynamic memory allocations are of the form `int * array = new int[ number_of_elements ]`.

You also saw that dynamic allocations using the `new[]` keyword require you to call `delete[]` on the array later, otherwise you'd have a memory leak. Having to manage memory this way is hard work.

Is there a way to create an array of dynamic size and have the memory automatically managed for you by C++? The answer is yes. There are C++ object types (commonly called containers) that handle dynamic memory allocations and deallocations automatically. UE4 provides a couple of container types to store your data in dynamically resizable collections.

There are two different groups of template containers. There is the UE4 family of containers (beginning with `T*`) and the C++ **Standard Template Library** (**STL**) family of containers. There are some differences between the UE4 containers and the C++ STL containers, but the differences are not major. UE4 containers sets are written with game performance in mind. C++ STL containers also perform well, and their interfaces are a little more consistent (consistency in an API is something that you'd prefer). Which container set you use is up to you. However, it is recommended that you use the UE4 container set since it guarantees that you won't have cross-platform issues when you try to compile your code.

# Debugging the output in UE4

All of the code in this chapter (as well as in the later chapters) will require you to work in a UE4 project. For the purpose of testing `TArray`, I created a basic code project called `TArrays`. In the `ATArraysGameMode::ATArraysGameMode` constructor, I am using the debug output feature to print text to the console.

Here's how the code will look:

[PRE0]

If you compile and run this project, you will see the debug text in the top-left corner of your game window when you start the game. You can use a debug output to see the internals of your program at any time. Just make sure that the `GEngine` object exists at the time of debugging the output. The output of the preceding code is shown in the following screenshot:

![Debugging the output in UE4](img/00133.jpeg)

# UE4's TArray<T>

TArrays are UE4's version of a dynamic array. To understand what a `TArray<T>` variable is, you first have to know what the `<T>` option between angle brackets stands for. The `<T>` option means that the type of data stored in the array is a variable. Do you want an array of `int`? Then create a `TArray<int>` variable. A `TArray` variable of `double`? Create a `TArray<double>` variable.

So, in general, wherever a `<T>` appears, you can plug in a C++ type of your choice. Let's move on and show this with an example.

## An example that uses TArray<T>

A `TArray<int>` variable is just an array of `int`. A `TArray<Player*>` variable will be an array of `Player*` pointers. An array is dynamically resizable, and elements can be added at the end of the array after its creation.

To create a `TArray<int>` variable, all you have to do is use the normal variable allocation syntax:

[PRE1]

Changes to the `TArray` variable are done using member functions. There are a couple of member functions that you can use on a `TArray` variable. The first member function that you need to know about is the way you add a value to the array, as shown in the following code:

[PRE2]

These four lines of code will produce the array value in memory, as shown in the following figure:

![An example that uses TArray<T>](img/00134.jpeg)

When you call `array.Add( number )`, the new number goes to the end of the array. Since we added the numbers **1**, **10**, **5**, and **20** to the array, in this order, that is the order in which they will go into the array.

If you want to insert a number in the front or middle of the array, it is also possible. All you have to do is use the `array.Insert(value, index)` function, as shown in the following line of code:

[PRE3]

This function will push the number **9** into the position **0** of the array (at the front). This means that the rest of the array elements will be offset to the right, as shown in the following figure:

![An example that uses TArray<T>](img/00135.jpeg)

We can insert another element into position **2** of the array using the following line of code:

[PRE4]

This function will rearrange the array as shown in the following figure:

![An example that uses TArray<T>](img/00136.jpeg)

### Tip

If you insert a number into a position in the array that is out of bounds, UE4 will crash. So be careful not to do that.

## Iterating a TArray

You can iterate (walk over) the elements of a `TArray` variable in two ways: either using integer-based indexing or using an iterator. I will show you both the ways here.

### The vanilla for loop and square brackets notation

Using integers to index the elements of an array is sometimes called a "vanilla" `for` loop. The elements of the array can be accessed using `array[ index ]`, where `index` is the numerical position of the element in the array:

[PRE5]

### Iterators

You can also use an iterator to walk over the elements of the array one by one, as shown in the following code:

[PRE6]

Iterators are pointers into the array. Iterators can be used to inspect or change values inside the array. An example of an iterator is shown in the following figure:

![Iterators](img/00137.jpeg)

The concept of an iterator: it is an external object that can look into and inspect the values of an array. Doing ++ it moves the iterator to examine the next element.

An iterator must be suitable for the collection it is walking through. To walk through a `TArray<int>` variable, you need a `TArray<int>::TIterator` type iterator.

We use `*` to look at the value behind an iterator. In the preceding code, we used `(*it)` to get the integer value from the iterator. This is called dereferencing. To dereference an iterator means to look at its value.

The `++it` operation that happens at the end of each iteration of the `for` loop increments the iterator, moving it on to point to the next element in the list.

Insert the code into the program and try it out now. Here's the example program we have created so far using `TArray` (all in the `ATArraysGameMode::ATArraysGameMode()` constructor):

[PRE7]

The output of the preceding code is shown in the following screenshot:

![Iterators](img/00138.jpeg)

## Finding whether an element is in the TArray

Searching out UE4 containers is easy. It is commonly done using the `Find` member function. Using the array we created previously, we can find the index of the value `10` by typing the following line of code:

[PRE8]

# TSet<T>

A `TSet<int>` variable stores a set of integers. A `TSet<FString>` variable. stores a set of strings. The main difference between `TSet` and `TArray` is that `TSet` does not allow duplicates—all the elements inside a `TSet` are guaranteed to be unique. A `TArray` variable does not mind duplicates of the same elements.

To add numbers to `TSet`, simply call `Add`. Take an example of the following declaration:

[PRE9]

This is how `TSet` will look, as shown in the following figure:

![TSet<T>](img/00139.jpeg)

Duplicate entries of the same value in the `TSet` will not be allowed. Notice how the entries in a `TSet` aren't numbered, as they were in a `TArray`: you can't use square brackets to access an entry in `TSet` arrays.

## Iterating a TSet

In order to look into a `TSet` array, you must use an iterator. You can't use square brackets notation to access the elements of a `TSet`:

[PRE10]

## Intersecting TSet

The `TSet`array has two special functions that the `TArray` variable does not. The intersection of two `TSet` arrays is basically the elements they have in common. If we have two `TSet` arrays such as `X` and `Y` and we intersect them, the result will be a third, new `TSet` array that contains only the elements common between them. Look at the following example:

[PRE11]

The common elements between `X` and `Y` will then just be the element `2`.

## Unioning TSet

Mathematically, the union of two sets is when you basically insert all the elements into the same set. Since we are talking about sets here, there won't be any duplicates.

If we take the `X` and `Y` sets from the previous example and create a union, we will get a new set, as follows:

[PRE12]

## Finding TSet

You can determine whether an element is inside a `TSet` or not by using the `Find()` member function on the set. The `TSet` will return a pointer to the entry in the `TSet` that matches your query if the element exists in the `TSet`, or it will return `NULL` if the element you're asking for does not exist in the `TSet`.

# TMap<T, S>

A `TMap<T, S>` creates a table of sorts in the RAM. A `TMap` represents a mapping of the keys at the left to the values on the right-hand side. You can visualize a `TMap` as a two-column table, with keys in the left column and values in the right column.

## A list of items for the player's inventory

For example, say we wanted to create a C++ data structure in order to store a list of items for the player's inventory. On the left-hand side of the table (the keys), we'd have an `FString` for the item's name. On the right-hand side (the values), we'd have an `int` for the quantity of that item.

| Item (Key) | Quantity (Value) |
| --- | --- |
| apples | 4 |
| donuts | 12 |
| swords | 1 |
| shields | 2 |

To do this in code, we'd simply use the following:

[PRE13]

Once you have created your `TMap`, you can access values inside the `TMap` using square brackets and by passing a key between the brackets. For example, in the `items` map in the preceding code, `items[ "apples" ]` is 4.

### Tip

UE4 will crash if you use square brackets to access a key that doesn't exist in the map yet, so be careful! The C++ STL does not crash if you do this.

## Iterating a TMap

In order to iterate a `TMap`, you use an iterator as well:

[PRE14]

`TMap` iterators are slightly different from `TArray` or `TSet` iterators. A `TMap` iterator contains both a `Key` and a `Value`. We can access the key inside with `it->Key` and the value inside the `TMap` with `it->Value`.

![Iterating a TMap](img/00140.jpeg)

# C++ STL versions of commonly used containers

I want to cover the C++ STL versions of a couple of containers. STL is the standard template library, which is shipped with most C++ compilers. The reason why I want to cover these STL versions is that they behave somewhat differently than the UE4 versions of the same containers. In some ways, their behavior is very good, but game programmers often complain of STL having performance issues. In particular, I want to cover STL's `set` and `map` containers.

### Note

If you like STL's interface but want better performance, there is a well-known reimplementation of the STL library by Electronic Arts called EASTL, which you can use. It provides the same functionality as STL but is implemented with better performance (basically by doing things such as eliminating bounds checking). It is available on GitHub at [https://github.com/paulhodge/EASTL](https://github.com/paulhodge/EASTL).

## C++ STL set

A C++ set is a bunch of items that are unique and sorted. The good feature about the STL `set` is that it keeps the set elements sorted. A quick and dirty way to sort a bunch of values is actually to just shove them into the same `set`. The `set` will take care of the sorting for you.

We can return to a simple C++ console application for the usage of sets. To use the C++ STL set you need to include `<set>`, as shown here:

[PRE15]

The following is the output of the preceding code:

[PRE16]

The duplicate `7` is filtered out, and the elements are kept in increasing order inside the `set`. The way we iterate over the elements of an STL container is similar to UE4's `TSet` array. The `intSet.begin()` function returns an iterator that points to the head of the `intSet`.

The condition to stop iterating is when `it` becomes `intSet.end()`. `intSet.end()` is actually one position past the end of the `set`, as shown in the following figure:

![C++ STL set](img/00141.jpeg)

### Finding an element in a <set>

To find an element inside an STL `set`, we can use the `find()` member function. If the item we're looking for turns up in the `set`, we get an iterator that points to the element we were searching for. If the item that we were looking for is not in the `set`, we get back `set.end()`instead, as shown here:

[PRE17]

### Exercise

Ask the user for a set of three unique names. Take each name in, one by one, and then print them in a sorted order. If the user repeats a name, then ask them for another one until you get to three.

### Solution

The solution of the preceding exercise can be found using the following code:

[PRE18]

## C++ STL map

The C++ STL `map` object is a lot like UE4's `TMap` object. The one thing it does that `TMap` does not is to maintain a sorted order inside the map as well. Sorting introduces an additional cost, but if you want your map to be sorted, opting for the STL version might be a good choice.

To use the C++ STL `map` object, we include `<map>`. In the following example program, we populate a map of items with some key-value pairs:

[PRE19]

This is the output of the preceding program:

[PRE20]

Notice how the iterator's syntax for an STL map is slightly different than that of `TMap`: we access the key using `it->first` and the value using `it->second`.

Notice how C++ STL also offers a bit of syntactic sugar over `TMap`; you can use square brackets to insert into the C++ STL `map`. You cannot use square brackets to insert into a `TMap`.

### Finding an element in a <map>

You can search a map for a <`key`, `value`> pair using the STL map's `find` member function.

### Exercise

Ask the user to enter five items and their quantities into an empty `map`. Print the results in sorted order.

### Solution

The solution of the preceding exercise uses the following code:

[PRE21]

In this solution code, we start by creating `map<string, int> items` to store all the items we're going to take in. Ask the user for an item and a quantity; then we save the `item` in the `items` map using square brackets notation.

# Summary

UE4's containers and the C++ STL family of containers are both excellent for storing game data. Often, a programming problem can be simplified many times by selecting the right type of data container.

In the next chapter, we will actually get to programming the beginning of our game by keeping track of what the player is carrying and storing that information in a `TMap` object.