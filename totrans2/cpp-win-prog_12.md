# Chapter 12. The Auxiliary Classes

Small Windows includes a set of auxiliary classes, which are as follows:

*   `Size`, `Point`, `Rect`, `Color`, and `Font`: These wrap the Win32 API structures which are `SIZE`, `POINT`, `RECT`, `COLORREF`, and `LOGFONT`. They are equipped with methods to communicate with files, the clipboard, and the registry. The Registry is a database in the Windows system that we can use to store values between the executions of our applications.
*   `Cursor`: is a type representing the Windows cursor.
*   `DynamicList`: holds a list of dynamic size with a set of callback functions.
*   `Tree`: holds a recursive tree structure.
*   `InfoList`: holds a list of generic information that can be transformed to and from a memory buffer.
*   There is also a small set of string manipulation functions.

# The Size class

The `Size` class is a small class holding the width and height:

**Size.h**

[PRE0]

The `ZeroSize` object is an object with its width and height set to zero:

[PRE1]

The default constructor initializes the width and height to zero. The size can be initialized by, and assigned to, another size. The `Size` class uses the assignment operator to assign a size to another size:

[PRE2]

A `Size` object can be initialized and assigned to a value of the Win32 API `SIZE` structure, and a `Size` object can be converted to a `SIZE`:

[PRE3]

When comparing two sizes, the widths are compared first. If they are equal, the heights are then compared:

[PRE4]

The multiplication operators multiply both the width and height with the factor. Note that even though the factor is a double, the resulting width and height are always rounded to integers:

[PRE5]

It is also possible to multiply the size with a pair of values, where the first value is multiplied by the width and the second value is multiplied by the height. Also, in this case, the resulting width and height are integers:

[PRE6]

The first set of addition operators adds and subtracts the distance to both the width and height:

[PRE7]

The second set of addition operators adds and subtracts the widths and heights separately:

[PRE8]

The size can be written to, and read from, a file stream, the clipboard, and the registry:

[PRE9]

The width and height are inspected by the constant methods and modified by the non-constant methods:

[PRE10]

The implementation of the `Size` class is rather straightforward:

**Size.cpp**

[PRE11]

As mentioned earlier, when comparing two sizes, the widths are compared first. If they are equal the heights are then compared:

[PRE12]

Note that `Min` and `Max` return the right-hand side value if the values are equal. We could let it return the left-hand side value instead. However, since the `Size` objects in that case hold the same *x* and *y* values and the methods return objects rather than references to an object, it does not matter. The same value is returned:

[PRE13]

As mentioned earlier, the resulting width and height are always rounded to integers, even though the factor is a double:

[PRE14]

When writing the size to the registry, we convert the size to a `SIZE` structure that is sent to `WriteBuffer` in `Registry`:

[PRE15]

When reading the size from the registry, we convert the default size to a `SIZE` structure that is sent to `ReadBuffer` in `Registry`. The result is then converted back to a `Size` object:

[PRE16]

# The Point class

The `Point` class is a small class holding the *x* and *y* position of a two-dimensional point:

**Point.h**

[PRE17]

The default constructor initializes the *x* and *y* value to zero. The point can be initialized by, and assigned to, another point:

[PRE18]

Similar to the `Size` class mentioned earlier, `Point` uses the assignment operator:

[PRE19]

Similar to `SIZE` in the preceding section, there is a `POINT` Win32 API structure. A `Point` object can be initialized by, and assigned to, a `POINT` structure, and a `Point` object can be converted to `POINT`:

[PRE20]

When comparing two points, the *x* values are first compared. If they are equal, the *y* values are then compared:

[PRE21]

Similar to the `Size` class mentioned earlier, the *x* and *y* values of the point can be multiplied by a factor. Note that even though the factor is a double, the resulting *x* and *y* values are always rounded to integers:

[PRE22]

It is also possible to multiply the point with a pair of values, where the first value is multiplied with the *x* value and the second value is multiplied with the *y* value. Also, in this case, the resulting *x* and *y* values are integers:

[PRE23]

The first set of addition operators adds and subtracts the integer distance to both the *x* and *y* value of the point:

[PRE24]

The second set of addition operators adds and subtracts the width and height of the size to the *x* and *y* values of the point:

[PRE25]

The third set of addition operators adds and subtracts the *x* and *y* values of the points:

[PRE26]

The point can be written to, and read from, a file stream, the clipboard, and the registry:

[PRE27]

The *x* and *y* value of the point are inspected by the constant methods and modified by the non-constant methods:

[PRE28]

The implementation of the `Point` class is also rather straightforward:

**Point.cpp**

[PRE29]

In the assignment operator, it is a good custom to verify that we do not assign the same object. However, it is not completely necessary in this case since we just assign the integer values of *x* and *y*:

[PRE30]

# The Rect class

The `Rect` class holds the four borders of a rectangle: left, top, right, and bottom.

**Rect.h**

[PRE31]

The default constructor sets all the four borders to zero. The rectangle can be initialized by, or assigned to, another rectangle. It is also possible to initialize the rectangle with the top-left and bottom-right corners, as well as the top-left corner and a size holding the width and height of the rectangle:

[PRE32]

Similar to `SIZE` and `POINT` in the previous sections, a rectangle can be initialized and assigned to a value of the Win32 API `RECT` structure. A `Rect` object can also be converted to a `RECT`:

[PRE33]

The compare operators first compare the top-left corners. If they are equal, the bottom-right corners are then compared:

[PRE34]

The multiplication operators multiply all sides with the factor. Even though the factor is a double, the border values are always integers, similar to the `Size` and `Point` cases of the previous sections:

[PRE35]

It is also possible to multiply the rectangle with a pair of values, where the first value is multiplied with `left` and `right`, and the second value is multiplied with `top` and `bottom`. Also, in this case, the resulting values are integers:

[PRE36]

The following operators are a little bit special: the addition operator adds the size to the bottom-right corner and leaves the top-left corner unchanged while the subtraction operator subtracts the size from the top-left corner and leaves the bottom-right corner unchanged:

[PRE37]

However, the following operators add and subtract the size to and from both the top-left and bottom-right corners:

[PRE38]

The following operators take a point as a parameter and add the point to, and subtract it from, both the top-left and bottom-right corner:

[PRE39]

The width of a rectangle is the absolute difference between the left and right border, and its height is the absolute difference between the top and bottom border:

[PRE40]

The `GetSize` method returns the width and height of the rectangle. It is not possible to name it `Size`, since there is a class with that name. However, it is still possible to define an operator returning a `Size` object. The `Size` and `Point` operators return the size and top-left corner of the rectangle:

[PRE41]

The top-left and bottom-right corner can both be inspected and modified. It is not appropriate to define methods returning a reference to a point since there are no corresponding fields for the corners:

[PRE42]

The `Clear` method sets all four corners to zero, `Normalize` swaps the left and right borders and the top and bottom borders if they appear in the wrong order, and `PointInside` returns `true` if the point is located inside the rectangle, assuming that it has been normalized:

[PRE43]

The rectangle can be written to and read from a file stream, the clipboard, and the registry:

[PRE44]

The four corners are inspected by the constant methods and modified by the non-constant methods:

[PRE45]

Similar to `Size` and `Point`, the implementation of `Rect` is rather straightforward.

**Rect.cpp**

[PRE46]

# The Color class

The `Color` class is a wrapper class for the Win32 API `COLORREF` structure, which holds a color in accordance with the Red-Green-Blue (RGB) standard. Each component of the color is represented by a value between 0 and 255, inclusive, which gives a theoretical total number of 256³ = 16,777,216 different colors, among which `Color` defines 142 standard colors.

**Color.h**

[PRE47]

The default constructor initializes the color with zero for each of the red, green, and blue values, which corresponds to black. A color object can also be initialized by, and assigned to, another color:

[PRE48]

The equality operators compare the red, green, and blue values:

[PRE49]

The `Inverse` function returns the inverted color and `GrayScale` returns the corresponding grayscale color:

[PRE50]

The color can be written to, and read from, a file stream, the clipboard, and the registry:

[PRE51]

The wrapped `COLORREF` structure value is inspected by the constant method and modified by the non-constant method:

[PRE52]

The predefined colors are constant objects:

[PRE53]

The implementation of `Color` is rather straightforward. The Win32 `RGB` macro creates a `COLORREF` value based on the three color components.

**Color.cpp**

[PRE54]

Two colors are equal if their wrapped `COLORREF` structures are equal, and they are compared with the C standard function `memcpy`.

[PRE55]

The `Inverse` function returns the inverted color with each component subtracted from 255, and `GrayScale` returns the corresponding grayscale color with each component holding the average value of the red, green, and blue components. `GetRValue`, `GetGValue`, and `GetBValue` are Win32 API macros that extract the red, green, and blue components:

[PRE56]

Each of the predefined colors calls the constructor that takes the red, green, and blue components:

[PRE57]

# The Font class

The `Font` class is a wrapper class for the Win32 API `LOGFONT` structure. The structure holds a large set of properties; however, we only take into consideration the fields for the font's name and size and whether the font is italic, bold, or underlined; the other fields are set to zero. The system font is the font where all fields in the `LOGFONT` structure are set to zero, which results in the standard font of the system. Finally, the `Font` class also includes a `Color` object.

**Font.h**

[PRE58]

The default constructor sets the name to the empty string and all other values to zero, resulting in the system font, usually 10 points Arial. The size of the font is given in typographic points (1 point = 1/72 of an inch = 1/72 * 25.4 mm ≈ 0.35 mm). A font can also be initialized by, or assigned to, another font:

[PRE59]

Two fonts are equal if they hold the same name and size as well as the same italic, bold, and underline status (all other fields are assumed to be zero):

[PRE60]

The font can be written to, and read from, a file stream, the clipboard, and the registry:

[PRE61]

The `PointToMeters` function converts a typographic point to logical units (hundredths of millimeters):

[PRE62]

The wrapped `LOGFONT` structure is inspected by the constant method and modified by the non-constant method:

[PRE63]

The `color` field can also be inspected by the constant method and modified by the non-constant method:

[PRE64]

**Font.cpp**

[PRE65]

Two fonts are equal if their wrapped `LOGFONT` structures and their `Color` fields are equal:

[PRE66]

The `write` and `read` methods write and read the wrapped `LOGFONT` structure and call the `Color` write and read methods:

[PRE67]

A typographic point is 1/72^(th) of an inch, and an inch is 25.4 millimeters. To transform a font typographical unit to logical units (hundredths of millimeters), we divide the width and height by 72, multiply by 2,540 (2,540 logical units equals 25.4 millimeters) and the zoom factor:

[PRE68]

# The Cursor class

There is a set of cursors available in the Win32 API, all with names starting with `IDC_`. In Small Windows, they have been given other names, which are hopefully easier to understand. Unlike other cases, we cannot use an enumeration for the cursors, since they are actually zero-terminated C++ strings (character pointers). Instead, every cursor is a pointer to a zero-terminated string. `LPCTSTR` stands for **Long Pointer to Constant TChar String**.

The reason the cursor has its own class, while the caret has a method in the `Document` class is that the caret does need a window handle to be set, while the cursor does not.

**Cursor.h**

[PRE69]

**Cursor.cpp**

[PRE70]

The `Set` method sets the cursor by calling the Win32 API functions `LoadCursor` and `SetCursor`:

[PRE71]

# The DynamicList class

The `DynamicList` class can be regarded as a more advanced version of the C++ standard classes `list` and `vector`. It varies its size dynamically:

**DynamicList.h**

[PRE72]

The `IfFuncPtr` pointer is a function prototype that is used when testing (without changing) a value in the list. It takes a constant value and a `void` pointer and returns a `Boolean` value. `DoFuncPtr` is used when changing a value in the list and takes a (non-constant) value and a `void` pointer. The void pointers are sent by the calling methods; they hold additional information:

[PRE73]

The list can be initialized by, and assigned to, another list. The default constructor creates an empty list, and the destructor deallocates the memory from the list:

[PRE74]

The `Empty` function returns `true` if the list is empty, `Size` returns the number of values in the list, `Clear` removes every value in the list, and `IndexOf` gives the zero-based index of the given value, or returns minus one if there is no such value in the list:

[PRE75]

The `begin` and `end` methods return pointers to the beginning and end of the list. They are included in order for the list to be iterated by the `for` statement:

[PRE76]

The index method inspects or modifies the value with the given zero-based index in the list:

[PRE77]

The `Front` and `Back` methods inspect and modify the first and the last value of the list by calling the index methods mentioned previously:

[PRE78]

The `PushFront` and `PushBack` methods add a value or a list at the beginning or at the end of the list, and `Insert` inserts a value or a list at the given index:

[PRE79]

The `Erase` function deletes the value at the given index, and `Remove` deletes the list from `firstIndex` to `lastIndex`, inclusive, or the end of the list if `lastIndex` is minus one. If `firstIndex` is zero and `lastIndex` is minus one, the whole list is deleted. The methods have been given different names since `lastIndex` in `Remove` is a default parameter. Giving the methods the same name would be a violation of the overload rules:

[PRE80]

The `Copy` function copies the list from `firstIndex` to `lastIndex`, inclusive, to `copyList` or the rest of the list if `lastIndex` is minus one, which implies that the whole list is copied if `firstIndex` is zero and `lastIndex` is minus one:

[PRE81]

The `AnyOf` function returns `true` if at least one value satisfies `ifFuncPtr`. That is, if `ifFuncPtr` returns `true` when called with the value as parameter. The `AllOf` function returns `true` if all values satisfy `ifFuncPtr`:

[PRE82]

The `FirstOf` and `LastOf` methods set the `value` parameter to the first and last value satisfying `ifFuncPtr`; they return `false` is there are no such values:

[PRE83]

The `Apply` method calls `doFuncPtr` for all values in the list, and `ApplyIf` calls `doFuncPtr` for each value in the list that satisfies `ifFuncPtr`:

[PRE84]

The `CopyIf` method copies each value in the list satisfying `ifFuncPtr` to `copyList`. `RemoveIf` removes the values satisfying `ifFuncPtr`:

[PRE85]

The `ApplyRemoveIf` method calls `doFuncPtr` to each value satisfying `ifFuncPtr` and then removes them. It may seem strange to apply a function to values that are to be removed. However, it is useful when removing dynamically allocated values, where `doFuncPtr` deallocates the memory of each value before it is removed from the list. It would not work to simply call `ApplyIf` and `RemoveIf`. When the values have been deleted by `ApplyIf`, they cannot be parameters to `ifFuncPtr` calls in `RemoveIf`:

[PRE86]

The size is the number of values in the list and the buffer holds the values themselves. The size of the buffer is dynamic and changes when values are added to, or removed from, the list. When the list is empty, the buffer points are null:

[PRE87]

The default constructor and assignment operator iterates through the given list and copies each value. For this to work, the type must support the assignment operator, which all types, except arrays, do:

[PRE88]

In the assignment operator, we first delete the buffer, as it may hold values. If the list is empty, the buffer points are null and the delete operator does nothing:

[PRE89]

The destructor simply deletes the buffer. Again, if the list is empty, the buffer points are null and the delete operator does nothing:

[PRE90]

The `Clear` method sets the size to zero and the buffer to null:

[PRE91]

The `IndexOf` method iterates through the list and returns the index of the found value, or it returns minus one if there is no such value:

[PRE92]

The `begin` method returns the address of the first value in the list:

[PRE93]

The `end` method returns the address one step beyond the last value in the list, which is the convention of list iterators in C++:

[PRE94]

An assertion occurs if the index is beyond the list:

[PRE95]

When adding a value at the end of the original list, we need to allocate a new list with one extra value and add the new value at the end:

[PRE96]

When adding a new list at the end of the original list, we need to allocate a new list with the size of the original and new lists, and copy the values from the original list to the new list:

[PRE97]

When inserting a new value at the beginning of the list, we need to copy all the values in the original list one step forward to make room for the new value:

[PRE98]

When inserting a new list, at the beginning of the list, we need to copy all its values and the number of steps corresponding to the size of the new list to make room for its values:

[PRE99]

We move the values of the original list in order to make room for the new list:

[PRE100]

When we have made room for the new list, we copy it to the original list at the beginning:

[PRE101]

The `Insert` method works in ways similar to `PushFront`. We need to allocate a new list and copy values in the original list to make room for the new values, and then copy the new values into the original list:

[PRE102]

When erasing a value in the list, we allocate a new smaller list and copy the remaining values to that list:

[PRE103]

First, we copy the values before the delete index:

[PRE104]

Then, we copy the values after the delete index:

[PRE105]

The `Remove` method works in the same way as `Delete`; the difference is that more than one value can be removed from the list; `removeSize` holds the number of values to be removed:

[PRE106]

The `Copy` method simply calls `PushBack` for each value to be copied:

[PRE107]

The `AnyOf` method iterates through the list and returns `true` if at least one value satisfies the function:

[PRE108]

The `AllOf` method iterates through the list and returns `false` if at least one value does not satisfy the function:

[PRE109]

The `FirstOf` method finds the first value in the list that satisfies the function, copies it to the value parameter, and returns `true`. If it does not find any value satisfying the function, `false` is returned:

[PRE110]

The `LastOf` method finds the last value satisfying the function in the same way as `FirstOf`; the difference is that the search is performed backward:

[PRE111]

The `Apply` method iterates through the list and calls `doFuncPtr` for each value, the value may be modified (actually, the point of `Apply` is that the value is modified) since the parameter to `doFuncPtr` is not constant:

[PRE112]

The `ApplyIf` method iterates through the list and calls `doFuncPtr` for each value that satisfies `ifFuncPtr`:

[PRE113]

The `CopyIf` method copies every value that satisfies `ifFuncPtr` to `copyList` by calling `PushBack`:

[PRE114]

The `RemoveIf` method removes every value that satisfies `ifFuncPtr` by calling `Delete` for each value:

[PRE115]

The `ApplyRemoveIf` method applies `doFuncPtr` to each value that satisfies `ifFuncPtr`. We cannot simply call `Apply` and `RemoveIf`, since `doFuncPtr` may deallocate the values in `Apply`, and `ifFuncPtr` in `RemoveIf` would not work when called on deleted values. Instead, we call `doFuncPtr` and call `Erase` immediately after. In this way, the values are not accessed after the call to `doFuncPtr`:

[PRE116]

# The Tree class

The C++ standard library hold a set of container classes for arrays, lists, vectors, sets, and maps. However, there is no class for a tree structure. Therefore, the `Tree` class has been added to Small Windows. A tree is made up of a set of nodes, among which, one is the root node. Each node holds a (possibly empty) list of child nodes:

**Tree.h**

[PRE117]

The tree can be written to, and read from, a file stream or the clipboard:

[PRE118]

Each tree node holds a value that is inspected by the constant method and modified by the non-constant method:

[PRE119]

The tree node also holds a list of child nodes, which is inspected by the constant method and modified by the non-constant method:

[PRE120]

The child list is an initializer list of tree nodes; it is empty by default:

[PRE121]

The default constructor and the assignment operator call `Init` to do the actual initialization of the tree:

[PRE122]

The destructor deletes the children recursively:

[PRE123]

The `WriteTreeToStream` method writes the node value and the number of children to the stream, and then calls itself recursively for each child:

[PRE124]

The `ReadTreeFromStream` method reads the node value and the number of children from the stream, creates the children, and calls itself recursively for each child:

[PRE125]

The `WriteTreeToClipboard` and `ReadTreeFromClipboard` methods work in ways similar to `WriteTreeToStream` and `ReadTreeFromStream`:

[PRE126]

# The InfoList class

The `InfoList` class is an auxiliary class with template methods that stores information in a character list; information can be added and extracted; or written to, or read from, a buffer.

**InfoList** **.h**

[PRE127]

The `Align` function increases the list one byte at a time until the size of the align type is a divisor of the list size:

[PRE128]

The `AddValue` function adds a value of the template type by adding its value byte by byte to the list, while `GetValue` gets the value at the beginning of the list by extracting it byte by byte from the list:

[PRE129]

The `AddString` function adds the characters of the text to the list along with a terminating zero character, while `GetString` reads the text from the list until it encounters the terminating zero character:

[PRE130]

**InfoList.cpp**

[PRE131]

The `FromBuffer` function adds each byte of the buffer to the list, while `ToBuffer` extracts and copies each byte of the list to the buffer:

[PRE132]

# Strings

There are a small set of string functions:

*   `CharPtrToGenericString`: This takes text as a `char` character pointer and returns the same text as a generic `String` object. Remember that the `String` class holds values of the `TCHAR` type, of which many are `char` or `wchar_t` depending on system settings.
*   `Split`: This takes a string and returns a list of strings holding the space-separated words of the text.
*   `IsNumeric`: This returns`true` if the text holds a numeric value.
*   `Trim`: This removes spaces at the beginning and at the end of the text.
*   `ReplaceAll`: This replaces one string with another string.
*   `WriteStringToStream` and `ReadStringFromStream`: These write and read a string to and from a stream.
*   `StartsWith` and `EndsWith`: These return`true` if the text starts or ends with the subtext.

**String.h**

[PRE133]

**String.cpp**

[PRE134]

The `IsNumeric` method uses the `IStringStream` method to read the value of the string and compare the number of characters read with the length of the text. If all the characters of the text are read, the text will hold a numeric value and `true` will be returned:

[PRE135]

# Summary

In this chapter, we studied the auxiliary classes used by Small Windows. In [Chapter 13](ch13.html "Chapter 13. The Registry, Clipboard, Standard Dialogs, and Print Preview"), *The Clipboard, Standard Dialogs, and Print Preview*, we will look into the registry, the clipboard, standard dialogs, and print preview.