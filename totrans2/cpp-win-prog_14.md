# Chapter 14. Dialogs, Controls, and Page Setup

In this chapter, we look into the implementation of the following:

*   **Custom dialogs**: The `Dialog` class is intended to be inherited by subclasses and equipped with controls.
*   **Controls**: The `Control` class and its subclasses. There are controls for edit fields, check boxes, radio buttons, list boxes, and combo boxes.
*   **Converters**: Between strings and other values. For instance, when the user inputs text that represents a numerical value, it is possible to add a converter that converts the text to a value, or gives an error message if the text does not hold a valid value.
*   **Page Setup**: Where we extend the `Dialog` class. The dialog is used when setting page settings for a document of the `StandardDocument` class. It handles information for headers, footers, and margins.

# Custom dialogs

The `Dialog` class handles a set of **controls**, which are added to the dialog by the `AddControl` method. For a subclass of the `Dialog` class, refer to `PageSetupDialog` in the last section of this chapter. The Dialog class provides a modal dialog, which means that all other windows in the application become disabled until the dialog is closed.

The user may navigate between controls with the ***Tab*** key and between radio buttons in the same group with the arrow keys. They can also use mnemonics to access controls.

**Dialog.h**

[PRE0]

The `dialogMap` field is used by `DialogProc` to look up the dialog receiving the messages:

[PRE1]

The `Dialog` class is a subclass of `Window` even though it calls the default `Window` constructor, which does not call the Win32 API function `CreateWindowEx`. Instead, `DoModal` collects information about the dialog and its controls and calls the Win32 API function `DialogBoxIndirectParam`:

[PRE2]

As the name implies, `DoModal` disables its parent window for as long as the dialog is visible. That is, until the user closes the dialog:

[PRE3]

The destructor deletes all controls, which implies that a subclass to `Dialog` should add dynamically allocated controls to the dialog without deleting them:

[PRE4]

The `AddControl` method assigns an identity number to the control and adds it to `idMap`.

[PRE5]

The `OnSize` function is called each time the user changes the size of the dialog, it iterates through the controls and adjusts their size so that they keep their size relative to the size of the dialog client area.

[PRE6]

When the user presses the ***Return*** key `OnReturn` is called, and when they press the ***Esc*** key `OnEscape` is called. Their default behavior is to close the dialog and return control to `DoModal` with 1 and 0 as the return code; 1 is interpreted as `true` and 0 as `false`.

[PRE7]

The `OnControlInit` method is intended to be overridden by subclasses and is called when the dialog is being initialized (when it receives the `WM_INITDIALOG` message).

[PRE8]

The `TryClose` method is intended to be overridden by subclasses and its default behavior is to return `true`. The `OnClose` method is called when the user tries to close the dialog, and its default behavior is to call `TryClose` and close the dialog if it returns `true`, in which case `OnDestroy` is also called:

[PRE9]

Each control is assigned an identity number when added to the dialog, which is mapped to a pointer to the control in `idMap`:

[PRE10]

The dialog has a header text, top-left position, font, regular style, and extended style, which are stored by the constructor and used by `DoModal` in the `DialogBoxIndirectParam` call. However, the size of the dialog is not a constructor parameter; instead, the size is based on the control dimensions:

[PRE11]

The `leftMargin`, `maxWidth`, `topMargin`, and `maxHeight` fields are used when calculating the size of the dialog. The idea is that its size will be adjusted so that the left and right margins as well as the top and bottom margins for the closest control are equal:

[PRE12]

The first control is not assigned the identity number of 0, since it will cause confusion when handling messages if the control with identity 0 is a push button. Instead, we initialize `currentId` with 1000, and decrease its value with each new control. It is necessary to decrease the value in order for the ***Tab*** key to work correctly in the dialog:

[PRE13]

When the dialog is initialized (by receiving the `WM_INITDIALOG` message), its size is stored in `originalClientSize` to be used by `OnSize` when calculating the size of the controls:

[PRE14]

The `DialogProc` method is called every time the dialog receives a message. Unlike `WindowProc`, it will return `TRUE` if the message has been handled and does not need further processing. Moreover, it will not call `DefWindowProc` at the end; instead it will return `FALSE` if the message has not been handled:

[PRE15]

**Dialog.cpp**

[PRE16]

The default dialog font is set to 12-point Times New Roman.

[PRE17]

The constructor calls the `Window` constructor, which sets the parent window pointer and does nothing else. That is, it does not call the Win32 API function `CreateWindowEx`. The `header`, `topLeft`, `style`, `extendedStyle`, and `font` fields are stored to be used by `DoModal`:

[PRE18]

The `DoModal` function makes the dialog enter the modal state. That is, its parent window becomes disabled until the dialog is destroyed. But, it first loads information to `infoList`. The `AddValue` method is a template method of the `InfoList` class and adds values of different types to the list:

[PRE19]

First, we need to add the value `1` in order to set the version of the dialog template we want to work with:

[PRE20]

The `0xFFFF` value indicates that we want to work with the extended dialog template:

[PRE21]

The next word is intended for a help identity; however, we do not use it so we just set it to 0:

[PRE22]

Then comes the extended and regular style. Besides the style sent to the constructor, we set the dialog to have a caption, a system menu, a modal frame, and a font. Due to the `DS_SETFONT` flag, we will later add information about the dialog font:

[PRE23]

The next value is the number of controls in the dialog, which is given by the size of `idMap`:

[PRE24]

The top-left position is given by the `topLeft` field:

[PRE25]

The size of the client area of the dialog is set by `maxWidth`, `leftMargin`, `maxHeight`, and `topMargin`, which has been calculated in `AddControl`. The width of the client area is the maximum width of the control set plus its left margin. In this way, we adjust the dialog to hold the controls with equal left and right margins as well as top and bottom margins to the closest control:

[PRE26]

The next two zeros indicate that we do not want to use a menu and that we use the default dialog `Windows` class:

[PRE27]

Then, we set the header of the dialog. The `AddString` method is an `InfoList` template method that adds the string with a terminating 0 to the information list:

[PRE28]

Finally, we set the font of the dialog. We extract the `LOGFONT` structure of the `Font` class and extract its size (`lfHeight`), whether it is bold (`lfWeight`) or italics, its character set (which is 0 since we do not use it), and the font name:

[PRE29]

When the dialog information has been added to the information list, we call `AddControlInfo` for each control in order for the control information to be added to the list:

[PRE30]

When the list has been fully loaded, we allocate a global buffer and load it with the list. The `ToBuffer` method copies the list into the buffer:

[PRE31]

We need the handle to the parent window, if present, and then we create the dialog by calling the Win32 API function `DialogBoxIndirectParam`, which will not return until the user closes the dialog. The last parameter is a pointer to the `Dialog` object that will be sent with the `WM_INITDIALOG` message. The return value stored in `result` is the second parameter to an `EndDialog` call:

[PRE32]

We return `true` if the result value does not equal 0:

[PRE33]

If the global buffer allocation does not succeed, we return `false`:

[PRE34]

The destructor iterates through `idMap` and deletes each control of the dialog:

[PRE35]

The `AddControl` method adds a control to the dialog. If it is the first control to be added (`idMap` is empty), `leftMargin` and `topMargin` are set to the top-left corner of the control, and `maxWidth` and `maxHeight` are set to the top-left corner plus the control width or height. However, if it is not, the first control we need to compare is its top-left corner and size, with the current values, in order to find the margins and maximum size of the control set:

[PRE36]

The identity number of the control is set to `currentId`, which is returned and decreased:

[PRE37]

The `OnSize` method compares the new size of the client area with its original size. The ratio between them is stored in `factorPair`:

[PRE38]

The controls of `idMap` are iterated and the original size of each control is multiplied with `factorPair`, the ratio between the new and original client area size. In this way, the control will keep their sizes relative to the size of the dialog client area when the user changes the dialog size.

[PRE39]

The `OnReturn` method is called when the user presses the ***Return*** key, `OnEscape` is called when they press the ***Esc*** key, and `OnClose` is called when they close the dialog. The default behavior is to call `TryClose` and, if it returns `true`, call the Win32 API function `EndDialog`, which causes the `DialogBoxIndirectParam` call in `DoModal` to return the integer value given as the second parameter to `EndDialog`:

[PRE40]

The `DialogProc` method is called each time the dialog receives a message. The first parameter is a handle to the dialog, which is mapped to a `Dialog` pointer by `dialogMap`:

[PRE41]

The `WM_INITDIALOG` case is called when the dialog is created, but before it becomes visible. When the dialog was created by the `DialogBoxIndirectParam` method, the last parameter was a pointer to the encapsulating `Dialog` object. That pointer is given in the `longParam` parameter, it is translated into a pointer to `Dialog`, and added to `dialogMap`:

[PRE42]

The Win32 API window handle of the dialog is assigned to `dialogHandle`, the original size of the client area is calculated and stored in `originalClientSize`, and `OnDialogInit` is called:

[PRE43]

For each control in the dialog, its window handle is set by calling the Win32 API function `GetDlgItem`, which takes the dialog window handle and the control identity number, set by `AddControl`. Similar to the original client size of the dialog, the original size and position of the controls are also stored. Finally, `OnControlInit` is called for each control:

[PRE44]

Since the message is handled, `TRUE` is returned:

[PRE45]

The `WM_SIZE` case is sent to the dialog each time its size has been changed. The width and height are stored in the lower and upper word of the `longParam` parameter. The `OnSize` method is called in order to handle the message:

[PRE46]

The `WM_CLOSE` case is called when the user tries to close the dialog. The `OnClose` method is called to handle the message, which may or may not close the dialog:

[PRE47]

The `WM_DESTROY` case is called when the dialog is being destroyed. Unlike `WM_CLOSE`, there is no way to prevent the dialog from being destroyed. Since `WM_DESTROY` is the last message sent to the dialog, the dialog is removed from `dialogMap`:

[PRE48]

The `WM_COMMAND` case is sent to the dialog when the user has performed some action with one of the controls. In cases where the action involves a control, its identity number is stored in the lower word of `wordParam`:

[PRE49]

If the identity number is `IDOK` or `IDCANCEL`, the user has pressed the ***Return*** or ***Esc*** key:

[PRE50]

If the identity number is not `IDOK` or `IDCANCEL`, we look up the control with `idMap` and the notification code in the higher word of `wordParam`. The notification code may have the same value as `IDOK` or `IDCANCEL`, which is why we use this somewhat cumbersome construction to handle the code:

[PRE51]

When a control gains or loses input focus, `OnGainFocus` or `OnLoseFocus` is called; when they change the input text of a text field, `OnChange` is called; when they change the selection of a combo box, list box, or multiple list box, `OnSelect` is called; and when they click on a push button, checkbox, or radio button, `OnClick` is called:

[PRE52]

When the command message has been handled, there is no need to further process it. Therefore, we return `true`:

[PRE53]

If the message has not been handled, we returns `false` in order for the message to be further processed by the Windows system:

[PRE54]

# Controls

Here is the Small Windows control hierarchy:

![Controls](img/B05475_14_01.jpg)

**Control.h**

[PRE55]

The constructor sends the parent window pointer to the `Window` constructer and stores the other values until it is added to the dialog information list by `AddControlInfo`:

[PRE56]

The following methods are intended to be overridden by subclasses and are by default empty:

[PRE57]

The rectangle holding the original size and position is set by `Dialog` when it receives the `MW_INITDIALOG` message:

[PRE58]

Each control has an identity number, given by `AddControl` in `Dialog`. It has a regular style; the extended style is always 0\. The style, top-left corner and control size, class name, and control text are added to the information list when `DoModal` in `Dialog` calls `AddControlInfo`:

[PRE59]

**Control.cpp**

[PRE60]

The constructor calls `AddControl` for its parent dialog to add the control to the dialog and to receive the control's identity number:

[PRE61]

The `AddControlInfo` method, which is called by `DoModal` in `Dialog`, adds the information of the control. First, we need to align the information list with the size of a double word (4 bytes):

[PRE62]

The help identity and extended style are always 0:

[PRE63]

The style is extended with the child and visible flags, indicating that the control is a child window of the dialog and that it becomes visible when the dialog becomes visible:

[PRE64]

The top-left corner and size of the control are given in **dialog** **units**, which are based on the dialog font and are translated into device units:

[PRE65]

The control identity number is given in order to identify the control when the user performs some action, such as clicking on a button or selecting a list item:

[PRE66]

Each control has a class name, which is button, list, combo, static (label), or edit (text field), and text, which is the text of a text field or the label of a box or button, but is ignored for list and combo boxes:

[PRE67]

Finally, it is possible to send extra data with the control. However, we pass on that opportunity and just send 0:

[PRE68]

## The button controls

There are four kinds of button controls: group box, push button, checkbox, and radio button. The checkbox and radio button can be checked; the `Check` and `IsChecked` methods are defined in `ButtonControl`.

**ButtonControl.h**

[PRE69]

**ButtonControl.cpp**

[PRE70]

We send the `BM_SETCHECK` message to a check, a checkbox, or a radio button and the `BM_GETCHECK` message to find out whether it is checked:

[PRE71]

A group box is quite simple; it encapsulates a set of other controls and has no functionality besides its graphical appearance.

**GroupBox.h**

[PRE72]

**GroupBox.cpp**

[PRE73]

The `clickListener` constructor parameter is a listener called when the user clicks on the button. The `OnClick` method is overridden from `Control`.

**PushButton.h**

[PRE74]

**PushButton.cpp**

[PRE75]

A checkbox works independently of other checkboxes. The `checkPtr` parameter is a pointer to a `Boolean` value set to `true` or `false`, depending on whether the checkbox is checked.

**CheckBox.h**

[PRE76]

**CheckBox.cpp**

[PRE77]

The `OnControlInit` method is overridden from `Control` and checks the box in accordance with the value that `checkPtr` points at. `OnClick` is also overridden from `Control` and sets the value to `true` if the box is checked:

[PRE78]

A radio button is intended to work in a group with other radio buttons, with exactly one button checked at the time. When the user checks one button in the group, it gets checked and the previously checked box get unchecked. Each radio button in the group has a zero-based index; `indexPtr` points to an integer value, common to all radio buttons in the group, which is set to the index of the button currently checked.

**RadioButton.h**

[PRE79]

**RadioButton.cpp**

[PRE80]

The constructor sends the group and tab stop styles to the `Control` constructor if the index is 0, since the first button is the first button in the group. All buttons in the group will not be accessed by the ***Tab*** key, but only the first button. The `group` style indicates that the button starts a group and all additional radio buttons are considered members of the group, until another button with the `group` style is added:

[PRE81]

The radio button is checked if it has the same index as the value that `indexPtr` points at, and the value is set to the index of the button that is checked:

[PRE82]

## List controls

There are two kinds of list box: single list box and multiple list box. The single list box selects exactly one item at a time, and the multiple list box selects one or several (or none at all) items at the same time. The constructor takes a string list that is loaded to the list box by `LoadList`.

**ListControl.h**

[PRE83]

**ListControl.cpp**

[PRE84]

The `LoadList` method adds the item text in `textList` to the (single or multiple) list box by calling the `LB_ADDSTRING` message:

[PRE85]

A (single) list box is a box holding a list of visible items, as opposed to a combo box where the items are dropped down. If necessary, the list can be scrolled. Only one item can be selected at a time, as opposed to the multiple list. Similar to the radio box group, the constructor takes the `indexPtr` pointer pointing at an integer value holding the zero-based index of the currently selected item. Moreover, the constructor also takes a string list that is loaded into the list box by `LoadList` in `ListControl`.

**ListBox.h**

[PRE86]

**ListBox.cpp**

[PRE87]

We send the `LB_SETCURSEL` message to select an item and `LB_GETCURSEL` to get the index of the currently selected item:

[PRE88]

A multiple list box is a list box where the user can select more than one value, or no value at all; therefore, the `indexSetPtr` parameter is a pointer to a set of indexes rather than a pointer to one index.

**MultipleListBox.h**

[PRE89]

**MultipleListBox.cpp**

[PRE90]

When the user selects 0 or several values in the multiple list, we iterate through the indexes and send the `LB_SETSEL` message for each index with a `Boolean` value indicating whether its item will be set:

[PRE91]

When checking which values are currently selected, we send the `LB_GETSEL` message for each index and add the indexes of the selected items to the set, which is then returned:

[PRE92]

## Combo box

A combo box is a drop-down list of items, from which the user can select one. The functionality of a combo box is equal to a list box, only their graphical appearance differs. Moreover, the functionality is also equivalent to a radio button group. Similar to `ListBox` and `Radiobutton`, the constructor takes the `indexPtr` parameter, which is a pointer to an integer value, holding the zero-based index of the item currently selected.

**ComboBox.h**

[PRE93]

**ComboBox.cpp**

[PRE94]

The `CB_ADDSTRING` message loads the combo box with items, `CB_SETCURSEL` sets the selected item, and `CB_GETCURSEL` returns the index of the selected item:

[PRE95]

## Label

A label is a displayed text that often serves as a prompt to a text field; it has no functionality besides its graphical appearance.

**Label.h**

[PRE96]

**Label.cpp**

[PRE97]

## The TextField class

The `TextField` class is a template for a text field; it takes the type of the value stored in the text field; an integer base for octal, decimal, or hexadecimal integers (ignored for non-integer types); and a converter of the `Converter` class in the next section, which converts between values and text. The constructor's `valuePtr` parameter is a pointer to the value to be edited in the text field.

**TextField.h**

[PRE98]

The `OnControlInit` method is called when the text field has been created. It converts the value to the text displayed in the text field. The `OnLoseFocus` method is called when the user leaves the text field and converts its text to a value of the template type if the text is valid. If it is not valid, the text field is set to the text converted from the latest valid value:

[PRE99]

The Win32 API function `GetWindowText` gets the text of the text field and `SetWindowText` sets its text. We need to convert from a zero-terminated character pointer string to a `String` object by calling the `String` constructor, and from a `String` object to a zero-terminated character pointer by calling the `c_str` method of the `String` class:

[PRE100]

When the text field has been initialized, the `ValueToText` method of the `Converter` class is called to convert the value pointed to by `valuePtr` to the text displayed in the text field:

[PRE101]

When the text field loses input focus, the text is evaluated by the `Check` method in order to decide whether it is suitable to be converted to a value. If it is suitable, the `ValueToText` method is called to do the actual converting, and then the text is loaded to the text field:

[PRE102]

# Converters

The `Converter` class is a template class intended to be specialized by type. Its task is to convert values between the template type and the `String` objects. The `Check` variable takes a string and returns `true` if it holds a valid value, `TextToValue` converts a text to a value, and `ValueToText` converts a value to a text.

**Converter.h**

[PRE103]

## Signed integers

Small Windows comes equipped with a set of predefined converters, which are specializations of `Converter`. One of these handles signed integer values of the type `int`.

**Converter.h**

[PRE104]

**Converter.cpp**

[PRE105]

When checking whether the given string holds a valid integer value, we create an `IStringStream` object (the generic version of the Standard C++ class `istringstream`, with `TCHAR` instead of `char`) initialized with the trimmed text (initial and terminating white spaces are removed). Then, we read the text into an integer variable with the base parameter and test whether the stream has reached end-of-file (`eof`). If it has, all characters of the text have been read, which implies that the text holds a valid integer value and `true` is returned:

[PRE106]

The conversion from a string to an integer is similar to `Check`, which we covered earlier, with the difference that we return the integer value assuming that `Check` has confirmed that the text holds a valid integer value:

[PRE107]

When converting an integer to a string, we use the `OStringStream` method (the generic version of `ostringstream`), write the value to the stream, and return the stream converted to a string by `str`:

[PRE108]

## Unsigned integers

Unsigned integers work in the same way as signed integers, the only difference is that `int` has been replaced by `unsigned` `int`:

**Converter.h**

[PRE109]

**Converter.cpp**

[PRE110]

## Double values

Double values ignore the base parameter and do not use the `setbase` manipulator; otherwise, the test and conversions work in the same way as in integer cases.

**Converter.h**

[PRE111]

**Converter.cpp**

[PRE112]

## Strings

The string case is trivial, since a string can always be converted to another string.

**Converter.h**

[PRE113]

## Rational numbers

A **rational** **number** is a number that can be expressed as a fraction of two integers, where the second integer is non-zero. We do not really use rational numbers in this section or complex numbers in the next section, in our applications. They are included only to demonstrate the converter, and they are implemented in the Appendix at the end of the book.

**Converter.h**

[PRE114]

When checking whether the text holds a valid rational number, we simply create an object of the `Rational` class. If the constructor accepts the text without throwing a `NotaRationalNumber` exception, we return `true`. If it throws the exception, the text is not acceptable and we return `false`.

**Converter.cpp**

[PRE115]

When converting a string to a rational number, we create and return a `Rational` object, assuming that `Check` has confirmed that the text holds a valid rational number:

[PRE116]

When converting a rational number to a string we call the `String` conversion operator of the `Rational` class.

[PRE117]

## Complex numbers

A complex number is the sum *z = x + yi* of a real number *x* and a real number *y* multiplied by the **imaginary** **unit** *i*, which is the solution of the equation *x* ² + 1 = 0\. The specialization of `Converter` with regard to the `Complex` class is similar to the `Rational` specialization.

**Converter.h**

[PRE118]

**Converter.cpp**

[PRE119]

# Page setup

The final section describes page setup functionality, divided into the `PageSetupInfo` class, which handles page setup information, the `PageSetupDialog`, which is a subclass of `Dialog` displayed for the user to input page setup information, and the `Template` function, which translates code input by the user in the **Page Setup** dialog to actual values.

## Page setup information

The `PageSetupInfo` class holds information about the page: portrait or landscape orientation, the margins, the text and font of the header and footer, whether the header and footer will be present on the first page, and whether the pages will be enclosed by a frame.

**PageSetupInfo.h**

[PRE120]

**PageSetupInfo.cpp**

[PRE121]

The default constructor initializes the default member values by calling `PageSetupInfo`.

[PRE122]

The default constructor and assignment operator copy the member values.

[PRE123]

The equality operators compare all the fields:

[PRE124]

Page setup information can be written to, or read from, a stream:

[PRE125]

## The Page Setup dialog

The `PageSetupDialog` class is a part of Small Windows and is displayed by the `StandardDocument` framework when the user selects the **Page Setup** menu item. The word processor earlier in this book gives an example. The `PageSetupDialog` class is a subclass of `Dialog` and provides the user with the possibility to input the information in `PageSetupInfo`. Note that the header and footer text can be annotated with blocks of code, explained in the next section.

![The Page Setup dialog](img/B05475_14_02.jpg)

**PageSetupDialog.h**

[PRE126]

Each push button has its own listener:

[PRE127]

The page setup information is pointed at by `infoPtr`, which is modified when the user changes the state of the controls. There is also `backupInfo`, in case the user cancels the dialog:

[PRE128]

**PageSetupDialog.cpp**

[PRE129]

The constructor sets the pointer `infoPtr` to point at the page setup information. The information is also stored in `backupInfo`, which will be used if the user cancels the dialog; refer to `OnCancel`:

[PRE130]

Each control gives the **Page Setup** dialog (`this`) as its parent dialog, which means that the controls will be deleted by the dialog's destructor. This implies that we do need to keep track of the controls in order to delete them manually. Actually, we will not delete them manually as it would result in dangling pointers:

[PRE131]

Note that we give a pointer as a reference for the value of the top margin. This value will be modified when the user changes the value:

[PRE132]

Similar to the `TextField` case, we give a pointer to a reference of the `HeaderFirst` value, which is a `Boolean` value. It will be modified when the user checks the box:

[PRE133]

The `OnHeaderFont` listener is called when the user presses the button:

[PRE134]

The `OnHeaderFont` and `OnFooterFont` methods display font dialogs:

[PRE135]

The `OnOk` and `OnCancel` methods terminate the dialog. The `OnCancel` method also copies the backup information that was stored by the constructor at the beginning, since no new information will be returned when the user cancels the dialog:

[PRE136]

## The Template function

When the user inputs text in the header and footer fields in the **Page Setup** dialog, they can insert code in the text, which needs to be translated into valid values. The code is shown in the following table:

| **Code** | **Description** | **Example** |
| %P | Path with suffix | `C:\Test\Test.wrd` |
| %p | Path without suffix | `C:\Test\Test` |
| %F | File with suffix | `Test.wrd` |
| %f | File without suffix | Test |
| %N | Total number of pages | 7 |
| %n | Current page | 5 |
| %c | Current Copy | 3 |
| %D | Date with full month | January 1, 2016 |
| %d | Date with abbreviated month | Jan 1, 2016 |
| %T | Time with seconds | 07:08:09 |
| %t | Time without seconds | 07:08 |
| %% | Percent character | % |

The task of the `Template` function is to replace the code with valid values. It takes the `templateText` string with template code and returns the text with the code replaced by valid values. It needs the current copy and page number as well as the total number of pages.

For instance, the `Page %n out of %N` text can be translated to **Page 3 out of 5** and `File: %F, date: %d` can be translated to **File: Text.txt, date: Dec 31, 2016**.

**Template.h**

[PRE137]

**Template.cpp**

[PRE138]

We start by replacing the `c`, `n`, and `N` code with the number of copies and the current and total pages. The numerical values are translated into strings by `to_String`:

[PRE139]

The file of the path is its text after the last backslash (**\**) and the suffix is its text after the last dot (**.**). If there is no backslash, the file is the same as the path; if there is no dot, the path and file without the suffix is the same as the file and path with the suffix:

[PRE140]

The current date and time are obtained by calling the Standard C functions `time` and `localtime_s`:

[PRE141]

The current time with and without seconds and the current date with whole and abbreviated month names are written to string output streams. The `setw` manipulator makes sure that two characters are always written, `setfill` fills with zeros if necessary, and `ios::right` writes the value in a right-aligned manner:

[PRE142]

Finally, we need to replace each instance of `%%` with `%`:

[PRE143]

# Summary

In this chapter, we looked into custom dialogs, controls, converters, and the Page Setup dialog. The only remaining part of the book is the implementation of the rational and complex classes.