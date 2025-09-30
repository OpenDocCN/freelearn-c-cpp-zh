# Chapter 4. Qt Core Essentials

> *This chapter will help you master Qt ways of basic data processing and storage. First of all, you will learn how to handle textual data and how to match text against regular expressions. Then, you will see how to store and fetch data from files and how to use different storage formats for text and binary data. By the end of this chapter, you will be able to implement non-trivial logic and data processing in your games efficiently. You will also know how to load external data in your games and how to save your own data in permanent storage for future use.*

# Text handling

Applications with a graphical user interface (and games surely fall into this category) are able to interact with users by displaying text and by expecting textual input from the user. We have already scratched the surface of this topic in the previous chapter by using the `QString` class. Now, we will go into more details.

## Manipulating strings

Text in Qt is internally encoded using Unicode, which allows to represent characters in almost all languages spoken in the world and is de facto standard for native encoding of text in most modern operating systems. You have to be aware though that contrary to the `QString` class, the C++ language does not use Unicode by default. Thus, each string literal (that is, each bare text you wrap in quotation marks) that you enter in your code needs to be converted to Unicode first before it can be stored in any of Qt's string handling classes. By default, this is done implicitly assuming that the string literal is UTF-8 encoded, but `QString` provides a number of static methods to convert from other encodings such as `QString::fromLatin1()` or `QString::fromUtf16()`. This conversion is done at runtime, which adds an overhead to the program execution time, especially if you tend to do a lot of such conversions in your programs. Luckily, there is a solution for this:

[PRE0]

You can wrap your string literal in a call to `QStringLiteral`, as shown in the preceding code, which if your compiler supports, will perform the conversion at compile time. It's a good habit to wrap all your string literals into `QStringLiteral` but it is not required, so don't worry if you forget to do that.

We will not go into great detail here when describing the `QString` class, as in many aspects it is similar to `std::string`, which is part of the standard C++. Instead, we will focus on the differences between the two classes.

### Encoding and decoding text

The first difference has already been mentioned—`QString` keeps the data encoded as Unicode. This has the advantage of being able to express text in virtually any language at the cost of having to convert from other encodings. Most popular encodings—UTF-8, UTF-16, and Latin1—have convenience methods in `QString` for converting from and to the internal representation. But, Qt knows how to handle many other encodings as well. This is done using the `QTextCodec` class.

### Tip

You can list the codecs supported on your installation by using the `QTextCodec::availableCodecs()`static method. In most installations, Qt can handle almost 1,000 different text codecs.

Most Qt entities that handle text can access instances of this class to transparently perform the conversion. If you want to perform such conversion manually, you can ask Qt for an instance of a codec by its name and make use of the `fromUnicode()` and `toUnicode()` methods:

[PRE1]

### Basic string operations

The most basic tasks that involve text strings are those where you add or remove characters from the string, concatenate strings, and access the string's content. In this regard, `QString` offers an interface that is compatible with `std::string`, but it also goes beyond that, exposing many more useful methods.

Adding data at the beginning or at the end of the string can be done using the `prepend()` and `append()` methods, which have a couple of overloads that accept different objects that can hold textual data, including the classic `const char*` array. Inserting data in the middle of a string can be done with the `insert()` method that takes the position of the character where we need to start inserting as its first argument and the actual text as its second argument. The `insert` method has exactly the same overloads as `prepend` and `append`, excluding `const char*`. Removing characters from a string is similar. The basic way to do this is to use the `remove()` method that accepts the position at which we need to delete characters and the number of characters to delete is as shown:

[PRE2]

There is also a remove overload that accepts another string. When called, all its occurrences are removed from the original string. This overload has an optional argument that states whether comparison should be done in the default case-sensitive (`Qt::CaseSensitive`) or case-insensitive (`Qt::CaseInsensitive`) way:

[PRE3]

To concatenate strings, you can either simply add two strings together or you can append one string to the other:

[PRE4]

Accessing strings can be divided into two use cases. The first is when you wish to extract a part of the string. For this, you can use one of these three methods: `left()`, `right()`, and `mid()` that return the given number of characters from the beginning or end of the string or extract a substring of a specified length, starting from a given position in the string:

[PRE5]

The second use case is when you wish to access a single character of the string. The use of the index operator works with `QString` in a similar fashion as with `std::string`, returning a copy or non-const reference to a given character that is represented by the `QChar` class, as shown in the following code:

[PRE6]

In addition to this, Qt offers a dedicated method—`at()`—that returns a copy of the character:

[PRE7]

### Tip

You should prefer to use `at()` instead of the index operator for operations that do not modify the character, as this explicitly sets the operation.

### The string search and lookup

The second group of functionality is related to searching for the string. You can use methods such as `startsWith()`, `endsWith()`, and `contains()` to search for substrings in the beginning or end or in an arbitrary place in the string. The number of occurrences of a substring in the string can be retrieved by using the `count()` method.

### Tip

Be careful, there is also a `count()` method that doesn't take any parameters and returns the number of characters in the string.

If you need to know the exact position of the match, you can use `indexOf()` or `lastIndexOf()` to receive the position in the string where the match occurs. The first call works by searching forward and the other one searches backwards. Each of these calls takes two optional parameters—the second one determines whether the search is case-sensitive (similar to how `remove` works). The first one is the position in the string where the search begins. It lets you find all the occurrences of a given substring:

[PRE8]

### Dissecting strings

There is one more group of useful string functionalities that makes `QString` different from `std::string`. That is, cutting strings into smaller parts and building larger strings from smaller pieces.

Very often, a string contains substrings that are glued together by a repeating separator. A common case is the **Comma-separated Values** (**CSV**) format where a data record is encoded in a single string where fields in the record are separated by commas. While you could extract each field from the record using functions that you already know (for example, `indexOf`), an easier way exists. `QString` contains a `split()` method that takes the separator string as its parameter and returns a list of strings that are represented in Qt by the `QStringList` class. Then, dissecting the record into separate fields is as easy as calling the following code:

[PRE9]

The inverse of this method is the `join()` method present in the `QStringList` class, which returns all the items in the list as a single string merged together with a given separator:

[PRE10]

### Converting between numbers and strings

`QString` also provides some methods for convenient conversion between textual and numerical values. Methods such as `toInt()`, `toDouble()`, or `toLongLong()` make it easy to extract numerical values from strings. Apart from `toDouble()`, they all take two optional parameters—the first one is a pointer to a `bool` variable that is set to `true` or `false` depending on whether the conversion was successful or not. The second parameter specifies the numerical base (for example, binary, octal, decimal, or hexadecimal) of the value. The `toDouble()` method only takes a `bool` pointer to mark the success or failure as shown in the following code:

[PRE11]

A static method called `number()` performs the conversion in the other direction—it takes a numerical value and number base and returns the textual representation of the value:

[PRE12]

If you have to combine both `QString` and `std::string` in one program, `QString` offers you the `toStdString()` and `fromStdString()` methods to perform an adequate conversion.

### Tip

Some of the other classes that represent values also provide conversions to and from `QString`. An example of such a class is `QDate`, which represents a date and provides the `fromString()` and `toString()` methods.

### Using arguments in strings

A common task is to have a string that needs to be dynamic in such a way that its content depends on the value of some external variable—for instance, you would like to inform the user about the number of files being copied, showing "copying file 1 of 2" or "copying file 2 of 5" depending on the value of counters that denote the current file and total number of files. It might be tempting to do this by assembling all the pieces together using one of the available approaches:

[PRE13]

There are a number of drawbacks to such an approach; the biggest of them is the problem of translating the string into other languages (this will be discussed later in this chapter) where in different languages their grammar might require the two arguments to be positioned differently than in English.

Instead, Qt allows us to specify positional parameters in strings and then replace them with real values. Positions in the string are marked with the `%` sign (for example, `%1`, `%2`, and so on) and they are replaced by making a call to `arg()` and passing it the value that is used to replace the next lowest marker in the string. Our file copy message construction code then becomes:

[PRE14]

The `arg` method can accept single characters, strings, integers, and real numbers and its syntax is similar to that of `QString::number()`.

## Regular expressions

Let's briefly talk about **regular expressions**—usually shortened as **regex** or **regexp**. You will need these regular expressions whenever you have to check whether a string or parts of it matches a given pattern or when you want to find specific parts inside the text and possibly want to extract them. Both the validity check and the finding/extraction are based on the so-called pattern of the regular expression, which describes the format a string must have to be valid, to be found, or to be extracted. Since this book is focused on Qt, there is unfortunately no time to cover regular expressions in depth. This is not a huge problem, however, since you can find plenty of good websites that provide introductions to regular expressions on the Internet. A short introduction can be found in Qt's documentation of `QRegExp` as well.

Even though there are many flavors of the regular expression's syntax, the one that Perl uses has become the *de facto* standard. According to `QRegularExpression`, Qt offers Perl-compatible regular expressions.

### Note

`QRegularExpression` was first introduced with Qt 5\. In the previous versions, you'll find the older `QRegExp` class. Since `QRegularExpression` is closer to the Perl standard and since its execution speed is much faster compared to `QRegExp`, we advise you to use `QRegularExpression` whenever possible. Nevertheless, you can read the `QRegExp` documentation about the general introduction of regular expressions.

# Time for action – a simple quiz game

To introduce you to the main usage of `QRegularExpression`, let's imagine this game: a photo, showing an object, is shown to multiple players and each of them has to estimate the object's weight. The player whose estimate is closest to the actual weight wins. The estimates will be submitted via `QLineEdit`. Since you can write anything in a line edit, we have to make sure that the content is valid.

So what does valid mean? In this example, we define that a value between 1 g and 999 kg is valid. Knowing this specification, we can construct a regular expression that will verify the format. The first part of the text is a number, which can be between 1 and 999\. Thus, the corresponding pattern looks like `[1-9][0-9]{0,2}`, where `[1-9]` allows—and demands—exactly one digit, except zero, which is optionally followed by up to two digits including zero. This is expressed through `[0-9]{0,2}`. The last part of the input is the weight's unit. With a pattern such as `(mg|g|kg)`, we allow the weight to be input in **milligrams** (**mg**), **grams** (**g**), or **kilograms** (**kg**). With `[ ]?`, we finally allow an optional space between the number and unit. Combined together, the pattern and construction of the related `QRegularExpression` object looks like this:

[PRE15]

## *What just happened?*

In the first line, we constructed the aforementioned `QRegularExpression` object while passing the regular expression's pattern as a parameter to the constructor. We also could have called `setPattern()` to set the pattern:

[PRE16]

Both the approaches are equivalent. If you have a closer look at the unit, you can see that right now, the unit is only allowed to be entered in lowercase. We want, however, to also allow it to be in uppercase or mixed case. To achieve this, we can of course write `(mg|mG|Mg|MG|g|G|kg|kG|Kg|KG)`. Not only is this a hell of a work when you have more units, this is also very error-prone, and so we opt for a cleaner and more readable solution. On the second line of the initial code example, you see the answer: a pattern option. We used `setPatternOptions()` to set the `QRegularExpression::CaseInsensitiveOption` option, which does not respect the case of the characters used. Of course, there are a few more options that you can read about in Qt's documentation on `QRegularExpression::PatternOption`. Instead of calling `setPatternOptions()`, we could have also passed the option as a second parameter to the constructor of `QRegularExpression`:

[PRE17]

Now, let's see how to use this expression to verify the validity of a string. For the sake of simplicity and better illustration, we simply declared a string called `input`:

[PRE18]

All we have to do is call `match()`, passing the string we would like to check against it. In return, we get an object of the `QRegularExpressionMatch` type that contains all the information that is further needed—and not only to check the validity. With `QRegularExpressionMatch::hasMatch()`, we then can determine whether the input matches our criteria, as it returns `true` if the pattern could be found. Otherwise, of course, `false` is returned.

Attentive readers surely will have noticed that our pattern is not quite finished. The `hasMatch()` method would also return `true` if we matched it against "foo 142g bar". So, we have to define that the pattern is checked from the beginning to the end of the matched string. This is done by the `\A` and `\z` anchors. The former marks the start of a string and the latter the end of a string. Don't forget to escape the slashes when you use such anchors. The correct pattern would then look as follows:

[PRE19]

## Extracting information out of a string

After we have checked that the sent guess is well formed, we have to extract the actual weight from the string. In order to be able to easily compare the different guesses, we further need to transform all values to a common reference unit. In this case, it should be a milligram, the lowest unit. So, let's see what `QRegularExpressionMatch` can offer us for this task.

With `capturedTexts()`, we get a string list of the pattern's captured groups. In our example, this list would contain "23kg" and "kg". The first element is always the string that was fully matched by the pattern followed by all the sub strings captured by the used brackets. Since we are missing the actual number, we have to alter the pattern's beginning to `([1-9][0-9]{0,2})`. Now, the list's second element is the number and the third element is the unit. Thus, we can write the following:

[PRE20]

In the function's first two lines, we set up the pattern and its option. Then, we match it against the passed argument. If `QRegularExpressionMatch::hasMatch()` returns `true`, the input is valid and we extract the number and unit. Instead of fetching the entire list of captured text with `capturedTexts()`, we query specific elements directly by calling `QRegularExpressionMatch::captured()`. The passed integer argument signifies the element's position inside the list. So, calling `captured(1)` returns the matched digits as a `QString`.

### Tip

`QRegularExpressionMatch::captured()` also takes `QString` as the argument's type. This is useful if you have used named groups inside the pattern, for example, if you have written `(?<number>[1-9][0-9]{0,2})`, then you can get the digits by calling `match.captured("number")`. Named groups pay off if you have long patterns or when there is a high probability that further brackets will be added in future. Be aware that adding a group at a later time will shift the indices of all the following groups by `1` and you will have to adjust your code!

To be able to calculate using the extracted number, we need to convert `QString` into an integer. This is done by calling `QString::toInt()`. The result of this conversion is then stored in the `weight` variable. Next, we fetch the unit and transform it to lowercase characters on-the-fly. This way, we can, for example, easily determine whether the user's guess is expressed in grams by checking the unit against the lowercase "g". We do not need to take care of the capital "G" or the variants "KG", "Kg", and the unusual "kG" for kilogram.

To get the standardized weight in milligrams, we multiply `weight` by 1,000 or 1,000,000, depending on whether this was expressed in g or kg. Lastly, we return this standardized weight. If the string wasn't well formed, we return `-1` to indicate that the given guess was invalid. It is then the caller's duty to determinate which player's guess was the best.

### Note

Pay attention to whether your chosen integer type can handle the weight's value. For our example, 100,000,000 is the biggest possible value that can be held by a signed integer on a 32-bit system. If you are not sure whether your code will be compiled on a 32-bit system, use `qint32`, which is guaranteed to be a 32-bit integer on every system that Qt supports, allowing decimal notations.

As an exercise, try to extend the example by allowing decimal numbers so that 23.5g is a valid guess. To achieve this, you have to alter the pattern in order to enter decimal numbers and you also have to deal with `double` instead of `int` for the standardized weight.

## Finding all pattern occurrences

Lastly, let's have a final look at how to find, for example, all numbers inside a string, even those leading with zeros:

[PRE21]

The `input` QString instance contains an exemplary text in which we would like to find all numbers. The "foo" as well as "1a" variables should not be found by the pattern since these are not valid numbers. Therefore, we set up the pattern defining that we require at least one digit, `[0-9]+`, and that this digit—or these digits—should be wrapped by word boundaries, `\b`. Note that you have to escape the slash. With this pattern, we initiate the `QRegularExpression` object and call `globalMatch()` on it. Inside the passed argument, the pattern will be searched. This time, we do not get `QRegularExpressionMatch` back but, instead, an iterator of the `QRegularExpressionMatchIterator` type. Since `QRegularExpressionMatchIterator` behaves like a Java iterator, with `hasNext()`, we check whether there is a further match and if so we bring up the next match by calling `next()`. The type of the returned match is then `QRegularExpressionMatch`, which you already know.

### Tip

If you need to know about the next match inside the `while` loop, you can use `QRegularExpressionMatchIterator::peekNext()` to receive it. The upside of this function is that it does not move the iterator.

This way, you can iterate all pattern occurrences in the string. This is helpful if you, for example, want to highlight a search string in text.

Our example would give the output: `("123"), ("09") and ("3")`.

Taking into account that this was just a brief introduction to regular expressions, we would like to encourage you to read the *Detailed Description* section in the documentation to `QRegularExpression`, `QRegularExpressionMatch`, and `QRegularExpressionMatchIterator`. Regular expressions are very powerful and useful, so, in your daily programming life, you can benefit from the profound knowledge of regular expressions!

# Data storage

When implementing games, you will often have to work with persistent data—you will need to store the saved game data, load maps, and so on. For that, you have to learn about the mechanisms that let you use the data stored on digital media.

## Files and devices

The most basic and low-level mechanism that is used to access data is to save and load it from the files. While you can use the classic file access approaches provided by C and C++, such as `stdio` or `iostream`, Qt provides its own wrapper over the file abstraction that hides platform-dependent details and provides a clean API that works across all platforms in a uniform manner.

The two basic classes that you will work with when using files are `QDir` and `QFile`. The former represents the contents of a directory, lets you traverse filesystems, creates and remove directories, and finally, access all files in a particular directory.

### Traversing directories

Traversing directories with `QDir` is really easy. The first thing to do is to have an instance of `QDir` in the first place. The easiest way to do this is to pass the directory path to the `QDir` constructor.

### Tip

Qt handles file paths in a platform-independent way. Even though the regular directory separator on Windows is a backwards slash character (`\`) and on other platforms it is the forward slash (`/`), Qt accepts forward slash as a directory separator on Windows platforms as well. Therefore, you can always use `/` to separate directories when you pass paths to Qt functions.

You can learn the native directory separator for the current platform is by calling the `QDir::separator()`static function. You can transform between native and non-native separators with the `QDir::toNativeSeparators()` and `QDir::fromNativeSeparators()`functions.

Qt provides a number of static methods to access some special directories. The following table lists these special directories and functions that access them:

| Access function | Directory |
| --- | --- |
| `QDir::current()` | The current working directory |
| `QDir::home()` | The home directory of the current user |
| `QDir::root()` | The root directory—usually `/` for Unix and `C:\` for Windows |
| `QDir::temp()` | The system temporary directory |

When you already have a valid `QDir` object, you can start moving between directories. To do that, you can use the `cd()` and `cdUp()` methods. The former moves to the named subdirectory, while the latter moves to the parent directory.

To list files and subdirectories in a particular directory, you can use the `entryList()` method, which returns a list of entries in the directory that match the criteria passed to `entryList()`. This method has two overloads. The basic version takes a list of flags that correspond to the different attributes that an entry needs to have to be included in the result and a set of flags that determine the order in which entries are included in the set. The other overload also accepts a list of file name patterns in the form of `QStringList` as its first parameter. The most commonly used filter and sort flags are listed as follows:

| Filter flags |
| --- |
| `QDir::Dirs, QDir::Files, QDir::Drives, QDir::AllEntries` | List directories, files, drives (or all) that match the filters |
| `QDir::AllDirs` | List all subdirectories regardless of whether they match the filter or not |
| `QDir::Readable, QDir::Writable, QDir::Executable` | List entries that can be read, written, or executed |
| `QDir::Hidden, QDir::System` | List hidden files and system files |
| **Sort flags** |
| `QDir::Unsorted` | The order of entries is undefined |
| `QDir::Name, QDir::Time, QDir::Size, QDir::Type` | Sort by appropriate entry attributes |
| `QDir::DirsFirst, QDir::DirsLast` | Determines whether directories should be listed before or after files |

Here is an example call that returns all JPEG files in the user's `home` directory sorted by size:

[PRE22]

### Tip

The `<<` operator is a nice and fast way to append entries to `QStringList`.

### Getting access to the basic file

Once you know the path to a file (either by using `QDir::entryList()`, from some external source, or even by hardcoding the file path in code), you can pass it to `QFile` to receive an object that acts as a handle to the file. Before the file contents can be accessed, the file needs to be opened using the `open()` method. The basic variant of this method takes a mode in which we need to open the file. The following table explains the modes that are available:

| Mode | Description |
| --- | --- |
| `ReadOnly` | This file can be read from |
| `WriteOnly` | This file can be written to |
| `ReadWrite` | This file can be read from and written to |
| `Append` | All data writes will be written at the end of the file |
| `Truncate` | If the file is present, its content is deleted before we open it |
| `Text` | Native line endings are transformed to `\n` and back |
| `Unbuffered` | The flag prevents the file from being buffered by the system |

The `open()` method returns `true` or `false` depending on whether the file was opened or not. The current status of the file can be checked by calling `isOpen()` on the file object. Once the file is open, it can be read from or written to depending on the options that are passed when the file is opened. Reading and writing is done using the `read()` and `write()` methods. These methods have a number of overloads, but I suggest that you focus on using those variants that accept or return a `QByteArray` object, which is essentially a series of bytes—it can hold both textual and nontextual data. If you are working with plain text, then a useful overload for `write` is the one that accepts the text directly as input. Just remember that the text has to be null or terminated. When reading from a file, Qt offers a number of other methods that might come in handy in some situations. One of these methods is `readLine()`, which tries to read from the file until it encounters a new line character. If you use it together with the `atEnd()` method that tells you whether you have reached the end of the file, you can realize the line-by-line reading of a text file:

[PRE23]

Another useful method is `readAll()`, which simply returns the file content, starting from the current position of the file pointer until the end of the file.

You have to remember though that when using these helper methods, you should be really careful if you don't know how much data the file contains. It might happen that when reading line by line or trying to read the whole file into memory in one step, you exhaust the amount of memory that is available for your process (you can check the size of the file by calling `size()` on the `QFile` instance). Instead, you should process the file's data in steps, reading only as much as you require at a time. This makes the code more complex but allows us to better manage the available resources. If you require constant access to some part of the file, you can use the `map()` and `unmap()` calls that add and remove mappings of the parts of a file to a memory address that you can then use like a regular array of bytes:

[PRE24]

### Devices

`QFile` is really a descendant class of `QIODevice`, which is a Qt interface that is used to abstract entities related to reading and writing. There are two types of devices: sequential and random access devices. `QFile` belongs to the latter group—it has the concepts of start, end, size, and current position that can be changed by the user with the `seek()` method. Sequential devices, such as sockets and pipes, represent streams of data—there is no way to rewind the stream or check its size; you can only keep reading the data sequentially—piece by piece, and you can check how far away you currently are from the end of data.

All I/O devices can be opened and closed. They all implement `open()`, `read()`, and `write()` interfaces. Writing to the device queues the data for writing; when the data is actually written, the `bytesWritten()` signal is emitted that carries the amount of data that was written to the device. If more data becomes available in the sequential device, it emits the `readyRead()` signal, which informs you that if you call `read` now, you can expect to receive some data from the device.

# Time for action – implementing a device to encrypt data

Let's implement a really simple device that encrypts or decrypts the data that is streamed through it using a very simple algorithm—the Caesar cipher. What it does is that when encrypting, it shifts each character in the plaintext by a number of characters defined by the key and does the reverse when decrypting. Thus, if the key is `2` and the plaintext character is `a`, the ciphertext becomes `c`. Decrypting `z` with the key `4` will yield the value `v`.

We will start by creating a new empty project and adding a class derived from `QIODevice`. The basic interface of the class is going to accept an integer key and set an underlying device that serves as the source or destination of data. This is all simple coding that you should already understand, so it shouldn't need any extra explanation, as shown:

[PRE25]

The next thing is to make sure that the device cannot be used if there is no device to operate on (that is, when `m_device == 0`). For this, we have to reimplement the `QIODevice::open()` method and return `false` when we want to prevent operating on our device:

[PRE26]

The method accepts the mode that the user wants to open the device with. We perform an additional check to verify that the base device was opened in the same mode before calling the base class implementation that will mark the device as open.

To have a fully functional device, we still need to implement the two protected pure virtual methods, which do the actual reading and writing. These methods are called by Qt from other methods of the class when needed. Let's start with `writeData()`, which accepts a pointer to a buffer containing the data and size of that a buffer:

[PRE27]

First, we copy the data into a local byte array. Then, we iterate the array, adding to each byte the value of the key (which effectively performs the encryption). Finally, we try to write the byte array to the underlying device. Before informing the caller about the amount of data that was really written, we emit a signal that carries the same information.

The last method that we need to implement is the one that performs decryption by reading from the base device and adding the key to each cell of the data. This is done by implementing `readData()`, which accepts a pointer to the buffer that the method needs to write to and the size of the buffer. The code is quite similar to that of `writeData()` except that we are subtracting the key value instead of adding it:

[PRE28]

First, we read from the underlying device as much as we can fit into the buffer and store the data in a byte array. Then, we iterate the array and set subsequent bytes of data buffer to the decrypted value. Finally, we return the amount of data that was really read.

A simple `main()` function that can test the class looks as follows:

[PRE29]

We use the `QBuffer` class that implements the `QIODevice` API and acts as an adapter for `QByteArray` or `QString`.

## *What just happened?*

We created an encryption object and set its key to `3`. We also told it to use a `QBuffer` instance to store the processed content. After opening it for writing, we sent some data to it that gets encrypted and written to the base device. Then, we created a similar device, passing the same buffer again as the base device, but now, we open the device for reading. This means that the base device contains ciphertext. After this, we read all data from the device, which results in reading data from the buffer, decrypting it, and returning the data so that it can be written to the debug console.

## Have a go hero – a GUI for the Caesar cipher

You can combine what you already know by implementing a full-blown GUI application that is able to encrypt or decrypt files using the Caesar cipher `QIODevice` class that we just implemented. Remember that `QFile` is also `QIODevice`, so you can pass its pointer directly to `setBaseDevice()`.

This is just a starting point for you. The `QIODevice` API is quite rich and contains numerous methods that are virtual, so you can reimplement them in subclasses.

## Text streams

Much of the data produced by computers nowadays is based on text. You can create such files using a mechanism that you already know—opening `QFile` to write, converting all data into strings using `QString::arg()`, optionally encoding strings using `QTextCodec`, and dumping the resulting bytes to the file by calling `write`. However, Qt provides a nice mechanism that does most of this automatically for you in a way similar to how the standard C++ `iostream` classes work. The `QTextStream` class operates on any `QIODevice` API in a stream-oriented way. You can send tokens to the stream using the `<<` operator, where they get converted into strings, separated by spaces, encoded using a codec of your choice, and written to the underlying device. It also works the other way round; using the `>>` operator, you can stream data from a text file, transparently converting it from strings to appropriate variable types. If the conversion fails, you can discover it by inspecting the result of the `status()` method—if you get `ReadPastEnd` or `ReadCorruptData`, then this means that the read has failed.

### Tip

While `QIODevice` is the main class that `QTextStream` operates on, it can also manipulate `QString` or `QByteArray`, which makes it useful for us to compose or parse strings.

Using `QTextStream` is simple—you just have to pass it the device that you want it to operate on and you're good to go. The stream accepts strings and numerical values:

[PRE30]

Apart from directing content into the stream, the stream can accept a number of manipulators, such as `endl`, which have a direct or indirect influence on how the stream behaves. For instance, you can tell the stream to display a number as decimal and another as hexadecimal with uppercase digits using the following code (highlighted in the code are all manipulators):

[PRE31]

This is not the end of the capabilities of `QTextStream`. It also allows us to display data in a tabular manner by defining column widths and alignments. Suppose that you have a set of records for game players that is defined by the following structure:

[PRE32]

Let's dump such info into a file in a tabular manner:

[PRE33]

After running the program, you should get a result similar to the one shown in the following screenshot:

![Text streams](img/8874OS_04_01.jpg)

One last thing about `QTextStream` is that it can operate on standard C file structures, which makes it possible for us to use `QTextStream` to, for example, write to `stdout` or read from `stdin`, as shown in the following code:

[PRE34]

## Data serialization

More than often, we have to store object data in a device-independent way so that it can be restored later, possibly on a different machine with a different data layout and so on. In computer science, this is called serialization. Qt provides several serialization mechanisms and now we will have a brief look at some of them.

### Binary streams

If you look at `QTextStream` from a distance, you will notice that what it really does is serialize and deserialize data to a text format. Its close cousin is the `QDataStream` class that handles serialization and deserialization of arbitrary data to a binary format. It uses a custom data format to store and retrieve data from `QIODevice` in a platform-independent way. It stores enough data so that a stream written on one platform can be successfully read on a different platform.

`QDataStream` is used in a similar fashion as `QTextStream`—the operators `<<` and `>>` are used to redirect data into or out of the stream. The class supports most of the built-in Qt types so that you can operate on classes such as `QColor`, `QPoint`, or `QStringList` directly:

[PRE35]

If you want to serialize custom data types, you can teach `QDataStream` to do that by implementing proper redirection operators.

# Time for action – serialization of a custom structure

Let's perform another small exercise by implementing functions that are required to use `QDataStream` to serialize the same simple structure that contains the player information that we used for text streaming:

[PRE36]

For this, two functions need to be implemented, both returning a `QDataStream` reference that was taken earlier as an argument to the call. Apart from the stream itself, the serialization operator accepts a constant reference to the class that is being saved. The most simple implementation just streams each member into the stream and returns the stream afterwards:

[PRE37]

Complementary to this, deserializing is done by implementing a redirection operator that accepts a mutable reference to the structure that is filled by data that is read from the stream:

[PRE38]

Again, at the end, the stream itself is returned.

## *What just happened?*

We provided two standalone functions that define redirection operators for the `Player` class to and from a `QDataStream` instance. This lets your class be serialized and deserialized using mechanisms offered and used by Qt.

## XML streams

XML has become one of the most popular standards that is used to store hierarchical data. Despite its verbosity and difficulty to read by human eye, it is used in virtually any domain where data persistency is required, as it is very easy to read by machines. Qt provides support for reading and writing XML documents in two modules. First, the `QtXml` module provides access using the **Document Object Model** (**DOM**) standard with classes such as `QDomDocument`, `QDomElement`, and others. We will not discuss this approach here, as now the recommended approach is to use streaming classes from the `QtCore` module. One of the downsides of `QDomDocument` is that it requires us to load the whole XML tree into the memory before parsing it. In some situations, this is compensated for by the ease of use of the DOM approach as compared to a streamed approach, so you can consider using it if you feel you have found the right task for it.

### Tip

If you want to use the DOM access to XML in Qt, remember to enable the `QtXml` module in your applications by adding a `QT += xml` line in the project configuration files.

As already said, we will focus on the stream approach implemented by the `QXmlStreamReader` and `QXmlStreamWriter` classes.

# Time for action – implementing an XML parser for player data

In this exercise, we are going to create a parser to fill data that represents players and their inventory in an RPG game:

[PRE39]

Save the following document somewhere. We will use it to test whether the parser can read it:

[PRE40]

Let's create a class called `PlayerInfoReader` that will wrap `QXmlStreamReader` and expose a parser interface for the `PlayerInfo` instances. The class will contain two private members—the reader itself and a `PlayerInfo` instance that acts as a container for the data that is currently being read. We'll provide a `result()` method that returns this object once the parsing is complete, as shown in the following code:

[PRE41]

The class constructor accepts a `QIODevice` pointer that the reader is going to use to retrieve data as it needs it. The constructor is trivial, as it simply passes the device to the `reader` object:

[PRE42]

Before we go into parsing, let's prepare some code to help us with the process. First, let's add an enumeration type to the class that will list all the possible tokens—tag names that we want to handle in the parser:

[PRE43]

To use these tags, we'll add a static method to the class that returns the token type based on its textual representation:

[PRE44]

You can notice that we are using a class called `QStringRef`. It represents a string reference—a substring in an existing string—and is implemented in a way that avoids expensive string construction; therefore, it is very fast. We're using this class here because that's how `QXmlStreamReader` reports tag names. Inside this static method, we are converting the string reference to a real string and trying to match it against a list of known tags. If the matching fails, `-1` is returned, which corresponds to our `T_Invalid` token.

Now, let's add an entry point to start the parsing process. Add a public `read` method that initializes the data structure and performs initial checks on the input stream:

[PRE45]

After clearing the data structure, we call `readNextStartElement()` on the reader to make it find the starting tag of the first element, and if it is found, we check whether the root tag of the document is what we expect it to be. If so, we call the `readPlayerInfo()` method and return its result, denoting whether the parsing was successful. Otherwise, we bail out, reporting an error.

The `QXmlStreamReader` subclasses usually follow the same pattern. Each parsing method first checks whether it operates on a tag that it expects to find. Then, it iterates all the starting elements, handling those it knows and ignoring all others. Such an approach lets us maintain forward compatibility, since all tags introduced in newer versions of the document are silently skipped by an older parser.

Now, let's implement the `readPlayerInfo` method:

[PRE46]

After verifying that we are working on a `PlayerInfo` tag, we iterate all the starting subelements of the current tag. For each of them, we check whether it is a `Player` tag and call `readPlayer()` to descend into the level of parsing data for a single player. Otherwise, we call `skipCurrentElement()`, which fast-forwards the stream until a matching ending element is encountered.

The structure of `readPlayer()` is similar; however, it is more complicated as we also want to read data from attributes of the `Player` tag itself. Let's take a look at the function piece by piece:

[PRE47]

After checking for the right tag, we get the list of attributes associated with the opening tag and ask for values of the two attributes that we are interested in. After this, we loop all child tags and fill the `Player` structure based on the tag names. By converting tag names to tokens, we can use a `switch` statement to neatly structure the code in order to extract information from different tag types, as shown in the following code:

[PRE48]

If we are interested in the textual content of the tag, we can use `readElementText()` to extract it. This method reads until it encounters the closing tag and returns the text contained within it. For the `Inventory` tag, we call the dedicated `readInventory()` method.

For the `Location` tag, the code is more complex than before as we again descend into reading child tags, extracting the required information and skipping all unknown tags:

[PRE49]

The last method is similar in structure to the previous one—iterate all the tags, skip everything that we don't want to handle (everything that is not an inventory item), fill the inventory item data structure, and append the item to the list of already parsed items, as shown in the following code:

[PRE50]

In `main()` of your project, write some code that will check whether the parser works correctly. You can use the `qDebug()` statements to output the sizes of lists and contents of variables. Take a look at the following code for an example:

[PRE51]

## *What just happened?*

The code you just wrote implements a full top-down parser of the XML data. First, the data goes through a tokenizer, which returns identifiers that are much easier to handle than strings. Then, each method can easily check whether the token it receives is an acceptable input for the current parsing stage. Based on the child token, the next parsing function is determined and the parser descends to a lower level until there is nowhere to descend to. Then, the flow goes back up one level and processes the next child. If at any point an unknown tag is found, it gets ignored. This approach supports a situation when a new version of software introduces new tags to the file format specification, but an old version of software can still read the file by skipping all the tags that it doesn't understand.

## Have a go hero – an XML serializer for player data

Now that you know how to parse XML data, you can create the complementary part—a module that will serialize `PlayerInfo` structures into XML documents using `QXmlStreamWriter`. Use methods such as `writeStartDocument()`, `writeStartElement()`, `writeCharacters()`, and `writeEndElement()` for this. Verify that the documents saved with your code can be parsed with what we implemented together.

## JSON files

**JSON** stands for **JavaScript Object Notation,** which is a popular lightweight textual format that is used to store object-oriented data in a human-readable form. It comes from JavaScript where it is the native format used to store object information; however, it is commonly used across many programming languages and a popular format for web data exchange. A simple JSON-formatted definition looks as follows:

[PRE52]

JSON can express two kinds of entities: objects (enclosed in braces) and arrays (enclosed in square brackets) where an object is defined as a set of key-value pairs, where a value can be a simple string, an object, or array. In the previous example, we had an object containing three properties—name, age, and inventory. The first two properties are simple values and the last property is an array that contains two objects with two properties each.

Qt can create and read JSON descriptions using the `QJsonDocument` class. A document can be created from the UTF-8-encoded text using the `QJsonDocument::fromJson()` static method and can later be stored in a textual form again using `toJson()`. Since the structure of JSON closely resembles that of `QVariant` (which can also hold key-value pairs using `QVariantMap` and arrays using `QVariantList`), conversion methods to this class also exist using a set of `fromVariant()` and `toVariant()` calls. Once a JSON document is created, you can check whether it represents an object or an array using one of the `isArray` and `isObject` calls. Then, the document can be transformed into `QJsonArray` or `QJsonObject` using the `toArray` and `toObject` methods.

`QJsonObject` is an iterable type that can be queried for a list of keys (using `keys()`) or asked for a value of a specific key (with a `value()` method). Values are represented using the `QJsonValue` class, which can store simple values, an array, or object. New properties can be added to the object using the `insert()` method that takes a key as a string, a value can be added as `QJsonValue`, and the existing properties can be removed using `remove()`.

`QJsonArray` is also an iterable type that contains a classic list API—it contains methods such as `append()`, `insert()`, `removeAt()`, `at()`, and `size()` to manipulate entries in the array, again working on `QJsonValue` as the item type.

# Time for action – the player data JSON serializer

Our next exercise is to create a serializer of the same `PlayerInfo` structure as we used for the XML exercise, but this time the destination data format is going to be JSON.

Start by creating a `PlayerInfoJSON` class and give it an interface similar to the one shown in the following code:

[PRE53]

All that is really required is to implement the `writePlayerInfo` method. This method will use `QJsonDocument::fromVariant()` to perform the serialization; thus, what we really have to do is convert our player data to a variant. Let's add a protected method to do that:

[PRE54]

Since the structure is really a list of players, we can iterate the list of players, serialize each player to a variant, and append the result to `QVariantList`. Having this function ready, we can descend a level and implement an overload for `toVariant()` that takes a `Player` object:

[PRE55]

### Tip

Qt's `foreach` macro takes two parameters—a declaration of a variable and a container to iterate. At each iteration, the macro assigns subsequent elements to the declared variable and executes the statement located directly after the macro. A C++11 equivalent of `foreach` is a range that is based for construct:

[PRE56]

This time, we are using `QVariantMap` as our base type, since we want to associate values with keys. For each key, we use the index operator to add entries to the map. The position key holds a `QPoint` value, which is supported natively by `QVariant`; however, such a variant can't be automatically encoded in JSON, so we convert the point to a variant map using the C++11 initializer list. The situation is different with the inventory—again, we have to write an overload for `toVariant` that will perform the conversion:

[PRE57]

The code is almost identical to the one handling `PlayerInfo` objects, so let's focus on the last overload of `toVariant`—the one that accepts `Item` instances:

[PRE58]

There is not much to comment here—we add all keys to the map, treating the item type as an integer for simplicity (this is not the best approach in a general case, as if we serialize our data and then change the order of values in the original enumeration, we will not get the proper item types after deserialization).

What remains is to use the code we have just written in the `writePlayerInfo` method:

[PRE59]

# Time for action – implementing a JSON parser

Let's extend the `PlayerInfoJSON` class and equip it with a reverse conversion:

[PRE60]

First, we read the document and check whether it is valid and holds the expected array. Upon failure, an empty structure is returned; otherwise, `readPlayerInfo` is called and is given `QJsonArray` to work with:

[PRE61]

Since the array is iterable, we can again use `foreach` to iterate it and use another method—`readPlayer`—to extract all the needed data:

[PRE62]

In this function, we used `QJsonObject::value()` to extract data from the object and then we used different functions to convert the data to the desired type. Note that in order to convert to `QPoint`, we first converted it to `QVariantMap` and then extracted the values before using them to build `QPoint`. In each case, if the conversion fails, we get a default value for that type (for example, an empty string). To read the inventory, we employ a custom method:

[PRE63]

What remains is to implement `readItem()`:

[PRE64]

## *What just happened?*

The class that was implemented can be used for bidirectional conversion between `Item` instances and a `QByteArray` object, which contains the object data in the JSON format. We didn't do any error checking here; instead, we relied on automatic type conversion handling in `QJsonObject` and `QVariant`.

## QSettings

While not strictly a serialization issue, the aspect of storing application settings is closely related to the described subject. A Qt solution for this is the `QSettings` class. By default, it uses different backends on different platforms, such as system registry on Windows or INI files on Linux. The basic use of `QSettings` is very easy—you just need to create the object and use `setValue()` and `value()` to store and load data from it:

[PRE65]

The only thing you need to remember is that it operates on `QVariant`, so the return value needs to be converted to the proper type if needed as shown in the last line of the preceding code. A call to `value()` can take an additional argument that contains the value to be returned if the requested key is not present in the map. This allows you to handle default values, for example, in a situation when the application is first started and the settings are not saved yet:

[PRE66]

The simplest scenario assumes that settings are "flat" in the way that all keys are defined on the same level. However, this does not have to be the case—correlated settings can be put into named groups. To operate on a group, you can use the `beginGroup()` and `endGroup()` calls:

[PRE67]

When using this syntax, you have to remember to end the group after you are done with it. An alternative to using the two mentioned methods is to pass the group name directly to invocation of `value()`:

[PRE68]

As was mentioned earlier, `QSettings` can use different backends on different platforms; however, we can have some influence on which is chosen and which options are passed to it by passing appropriate options to the constructor of the `settings` object. By default, the place where the settings for an application are stored is determined by two values—the organization and the application name. Both are textual values and both can be passed as arguments to the `QSettings` constructor or defined a priori using appropriate static methods in `QCoreApplication`:

[PRE69]

This code is equivalent to:

[PRE70]

All of the preceding code use the default backend for the system. However, it is often desirable to use a different backend. This can be done using the `Format` argument, where we can pass one of the two options—`NativeFormat` or `IniFormat`. The former chooses the default backend, while the latter forces the INI-file backend. When choosing the backend, you can also decide whether settings should be saved in a system-wide location or in the user's settings storage by passing one more argument—the scope of which can be either `UserScope` or `SystemScope`. This can extend our final construction call to:

[PRE71]

There is one more option available for total control of where the settings data resides—tell the constructor directly where the data should be located:

[PRE72]

### Tip

The `QStandardPaths` class provides methods to determine standard locations for files depending on the task at hand.

`QSettings` also allows you to register your own formats so that you can control the way your settings are stored—for example, by storing them using XML or by adding on-the-fly encryption. This is done using `QSettings::registerFormat()`, where you need to pass the file extension and two pointers to functions that perform reading and writing of the settings, respectively, as follows:

[PRE73]

## Pop quiz – Qt core essentials

Q1\. What is the closest equivalent `std::string` in Qt?

1.  `QString`
2.  `QByteArray`
3.  `QStringLiteral`

Q2\. Which regular expression can be used to validate an IPv4 address, which is an address composed of four dot-separated decimal numbers with values ranging from 0 to 255?

Q3\. Which do you think is the best serialization mechanism to use if you expect the data structure to evolve (gain new information) in future versions of the software?

1.  JSON
2.  XML
3.  QDataStream

# Summary

In this chapter, you learned a number of core Qt technologies ranging from text manipulation, to accessing devices that can be used to transfer or store data using a number of popular technologies such as XML or JSON. You should be aware that we have barely scratched the surface of what Qt offers and there are many other interesting classes you should familiarize yourself with but this minimum amount of information should give you a head start and show you the direction to follow with your future research.

In the next chapter, we will switch from describing data manipulation, which can be visualized using text or only in your imagination, to a more appealing media. We will start talking about graphics and how to transfer what you can see in your imagination to the screen of your computer.