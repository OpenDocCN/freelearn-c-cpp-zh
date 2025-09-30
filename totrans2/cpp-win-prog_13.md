# Chapter 13. The Registry, Clipboard, Standard Dialogs, and Print Preview

This chapter describes the implementation of:

*   **The Registry:** A Windows database holding information between application executions.
*   **The Clipboard:** A Windows database holding information that has been cut, copied, and pasted.
*   **The standard dialogs**: This is used for saving and opening documents, for colors and fonts, and for printing.
*   **Print preview:** In the `StandardDocument` class, it is possible to view the document on the screen as if it is being printed.

# The registry

The static write, read, and erase methods in the `Registry` class operate on values of the `Integer`, `Double`, `Boolean`, and `String` types, as well as memory blocks in the Windows Registry.

**Registry.h**:

[PRE0]

**Registry.cpp**:

[PRE1]

The global constant `RegistryFileName` holds the path to the registry domain of Small Windows:

[PRE2]

The `WriteInteger`, `WriteDouble`, and `WriteBoolean` functions simply convert the value to a string and call `WriteString`:

[PRE3]

The `WriteString` function calls the Win32 API function `WritePrivateProfileString`, which writes the string to the registry. All the C++ `String` objects need to be converted to zero-terminated C strings (char pointers) by `c_str`:

[PRE4]

The `WriteBuffer` function calls the Win32 API function `WritePrivateProfileStruct`, which writes the memory block to the registry:

[PRE5]

The `ReadInteger`, `ReadDouble`, and `ReadBoolean` functions convert the default value to a string and call `ReadString`. The return value of `ReadString` is then converted and returned; `_tstoi` and `_tstof` are the generic versions of the standard C functions `atoi` and `atof`:

[PRE6]

The `ReadString` function calls the Win32 API function `GetPrivateProfileString`, which reads the string value to `text` and returns the number of characters read. If the number of read characters is greater than zero, the text is converted to a `string` object and returned; otherwise, the default text is returned:

[PRE7]

The `ReadBuffer` function calls the Win32 API function `ReadPrivateProfileStruct`, which reads the memory block from the registry. If it returns zero, it means that the reading failed and the default buffer is copied to the buffer:

[PRE8]

When erasing a value from the registry, we call `WritePrivateProfileString` with a null pointer instead of a string, which erases the value:

[PRE9]

# The Clipboard class

The `Clipboard` class is an interface to the global Windows Clipboard, which makes it possible to cut, copy, and paste information between different kinds of applications. There are two forms of clipboard operations: ASCII and Unicode text and generic (application-specific) information.

**Clipboard.h**:

[PRE10]

The formats for ASCII and Unicode lines are predefined.

[PRE11]

`Open` and `Close` open and close the clipboard. They return `true` if they succeed. `Clear` clears the clipboard when it has been opened. More specifically, it removes any potential information with the specified format and `Available` returns `true` if there is information with the format stored on the clipboard.

Information in different formats may be stored on the clipboard. For instance, when the user copies text in an application, the text may be stored on the clipboard as ASCII and Unicode text, as well as a more advanced application-specific format. `Available` returns `true` if information is stored on the clipboard with the specified format:

[PRE12]

The `WriteText` and `ReadText` functions write and read a list of strings, while the `WriteGeneric` and `ReadGeneric` functions write and read generic information:

[PRE13]

**Clipboard.cpp**:

[PRE14]

The `Open`, `Close`, and `Clear` functions call the Win32 API functions `OpenClipboard`, `CloseClipboard`, and `EmptyClipboard`. They all return integer values; a non-zero value indicates success:

[PRE15]

The `Available` function examines whether there is data with the format available on the clipboard by calling the Win32 API function `FormatAvailable`:

[PRE16]

## ASCII and Unicode lines

As `WriteText` and `ReadText` are template methods, they are included in the header file instead of the implementation file. `WriteText` takes a list of generic strings and writes them in any format to the clipboard; `AsciiFormat` (one byte/character) and `UnicodeFormat` (two bytes/character) are predefined.

**Clipboard.h**:

[PRE17]

First, we need to find the buffer size, which we calculate by adding the total number of characters in the lines. We also add one for each line since each line also holds a terminating character. The terminating character is the return character (`\r`) for each line, except the last line, which is terminated by a zero character (`\0`):

[PRE18]

When we have calculated the buffer size, we can call the Win32 API `GlobalAlloc` function to allocate the buffer in the global clipboard. We will later connect it to the format. We use the size of the template character type for the buffer:

[PRE19]

If the allocation succeeds, we receive a handle to the buffer. Since the clipboard and its buffers can be used by several processes at the same time, we need to lock the buffer by calling the Win32 API function `GlobalLock`. As long as the buffer is locked, no other processes can access it. When we lock the buffer we receive a pointer to it, which we can use when writing information to the buffer:

[PRE20]

We write the characters of the line to the buffer, and we add a `return` character unless it is the last line in the list:

[PRE21]

We add a zero character at the end of the buffer to mark its ending:

[PRE22]

When the buffer has been loaded with information, we only need to unlock the buffer so that other processes can access it and associate the buffer with the format:

[PRE23]

Finally, we return `true` to indicate that the operation succeeded:

[PRE24]

If we were not able to allocate a buffer to write the line list to, we indicate that the operation did not succeeded by returning `false`:

[PRE25]

When reading the line list with `ReadText`, we use `Format` (which usually is `AsciiFormat` or `UnicodeFormat`) to receive a handle from the clipboard, which we then use to lock the buffer and receive its pointer, which in turn allows to us read from the buffer:

[PRE26]

Note that we have to divide the buffer size with the template character type size (which may be greater than 1) in order to find the number of characters:

[PRE27]

When we encounter a return character (`\r`), the current line is finished; we add it to the line list and then clear it in order for it to be ready for the next line:

[PRE28]

When we encounter a return character (`'\0'`), we also add the current line to the line list. However, there is no need to clear the current line, since the zero character is the last character in the buffer:

[PRE29]

If the character is neither a return nor a zero character, we add it to the current line. Note that we read a character of the template `CharType` type and convert it to a generic character of the `TCHAR` type:

[PRE30]

Finally, we unlock the buffer and return `true` to indicate that the operation succeeded:

[PRE31]

If we do not receive a buffer for the format, we return `false` to indicate that the operation did not succeed:

[PRE32]

## Generic information

The `WriteGeneric` function is actually simpler than the preceding `WriteText` function, since it does need to take line lists into consideration. We simply lock the clipboard buffer, write each byte in `infoList` to the buffer, unlock the buffer, and associate it with the format.

**Clipboard.cpp**:

[PRE33]

The `ToBuffer` object in the `InfoList` function writes its bytes to the buffer:

[PRE34]

If we do not manage to allocate the global buffer, we return `false` to indicate that the operation did not succeed:

[PRE35]

The `ReadGeneric` function locks the clipboard buffer, writes each byte in the buffer to `infoList`, unlocks the buffer, and returns `true` to indicate that the operation succeeded:

[PRE36]

If we do not receive the global handle, we return `false` to indicate that the operation did not succeed:

[PRE37]

# Standard dialogs

In Windows, it's possible to define **dialogs**. Unlike windows, dialogs are intended to be populated with controls such as buttons, boxes, and text fields. A dialog may be **modal**, which means that the other windows of the application become disabled until the dialog is closed. In the next chapter, we will look into how we build our own dialogs.

However, in this section, we will look into the Windows **standard** **dialogs** for saving and opening files, choosing fonts and colors, and printing. Small Windows supports standard dialogs by wrapping the Win32 API function, which provides us with the dialogs.

## The Save dialog

The `SaveDialog` function displays the standard **Save** dialogs.

![The Save dialog](img/B05475_13_01.jpg)

The `filter` parameters filter the file types to be displayed. Each file format is defined in two parts: the text displayed in the dialog and the default file suffix. The parts are separated by a zero character and the filter is terminated with two zero characters. For instance, consider the following:

[PRE38]

The `fileSuffixList` parameter gives the allowed file suffixes and `saveFlags` holds the flags of the operation. The following two flags are available:

*   `PromptBeforeOverwrite`: This flag is a warning message that is displayed if the file does already exist
*   `PathMustExist`: This flag is an error message that is displayed if the path does not exist

**StandardDialog.h**:

[PRE39]

**StandardDialog.cpp**:

[PRE40]

The Win32 API `OPENFILENAME` structure `saveFileName` is loaded with appropriate values: `hwndOwner` is set to the window's handle, `hInstance` is set to the application instance handle, `lpstrFilter` is set to the `filter` parameter, `lpstrFile` is set to `pathBuffer`, which in turn holds the `path` parameter, and `Flags` is set to the `saveFlags` parameter:

[PRE41]

When `saveFileName` is loaded, we call the Win32 API function `GetSaveFileName`, which displays the standard **Save** dialog and returns a non-zero value if the user terminates the dialog by clicking on the **Save** button or pressing the **Return** key. In that case, we set the `path` parameter to the chosen path, check whether the path ends with one of the suffixes in `fileSuffixList`, and return `true` if it does. If the path suffix is not present in the list, we display an error message and the saving process starts over again. If the user cancels the process, `false` is returned. In fact, the only way for the user to finish the process is to choose a file with a suffix in the list or to cancel the dialog:

[PRE42]

## The Open dialog

The `OpenDialog` function displays the standard **Open** dialog.

![The Open dialog](img/B05475_13_02.jpg)

The `filter` and `fileSuffixList` parameters work in the same way as in the preceding `SaveDialog` function. There are three flags available:

*   `PromptBeforeCreate`: This flag displays a warning message if the file already exists
*   `FileMustExist`: The opened file must exist
*   `HideReadOnly`: This flag indicates that read-only files are hidden in the dialog

**OpenDialog.h**:

[PRE43]

The implementation of `OpenDialog` is similar to the preceding `SaveDialog` function. We use the same `OPENFILENAME` structure; the only difference is that we call `GetOpenFileName` instead of `GetSaveFileName`.

**OpenDialog.cpp**:

[PRE44]

## The Color dialog

The `ColorDialog` function displays a standard **Color** dialog.

![The Color dialog](img/B05475_13_03.jpg)

**StandardDialog.h**:

[PRE45]

The static `COLORREF` array `customColorArray` is used by the user in the color dialog to store the chosen colors. Since it is static, the `customColorArray` array is reused between dialog display sessions.

The `ColorDialog` function uses the Win32 API `CHOOSECOLOR` structure to initialize the dialog. The `hwndOwner` function is set to the window's handle, `rgbResult` is set to the color's `COLORREF` field, and `lpCustColors` is set to the custom color array. The `CC_RGBINIT` and `CC_FULLOPEN` flags initialize the dialog with the given color so that it is fully extended.

**StandardDialog.cpp**:

[PRE46]

The Win32 `ChooseColor` function displays the **Color** dialog and returns a non-zero value if the user terminates the dialog by clicking on the **OK** button. In that case, we set the chosen color and return `true`:

[PRE47]

If the user cancels the dialog, we return `false`:

[PRE48]

## The Font dialog

The `FontDialog` function displays a standard **Font** dialog.

![The Font dialog](img/B05475_13_04.jpg)

**StandardDialog.h**:

[PRE49]

**FontDialog.cpp**:

[PRE50]

The Win32 API `CHOOSEFONT` structure `chooseFont` is loaded with appropriate values. The `lpLogFont` object is set to the font's `LOGFONT` field and `rgbColors` is set to the color's `COLORREF` field:

[PRE51]

The Win32 `ChooseFont` function displays the **Font** dialog and returns a non-zero value if the user clicks on the **OK** button. In that case, we set the chosen font and color and return `true`:

[PRE52]

If the user cancels the dialog, we return `false`:

[PRE53]

## The Print dialog

The `PrintDialog` function displays a standard **Print** dialog.

![The Print dialog](img/B05475_13_05.jpg)

If the user clicks on the **Print** button, the chosen print settings are saved in the `PrintDialog` parameters:

**PrintDialog.h**:

[PRE54]

The `PrintDialog` function loads the Win32 API `PRINTDLG` structure `printDialog` with appropriate values, `nFromPage` and `nToPage` are set to the first and last page to be printed (whose default values are 1 and the number of pages respectively), `nMaxPage` is set to the number of pages, and `nCopies` is set to 1 (the default value).

**PrintDialog.cpp**:

[PRE55]

The Win32 API function `PrintDlg` displays the standard print dialog and returns a non-zero value if the user finishes the dialog by pressing the **Print** button. In that case, the first and last page to be printed, the number of copies, and whether the copies will be sorted are stored in the parameters, and the pointer to the `Graphics` object to be used when printing is created and returned.

If the user has chosen a page interval, we use the `nFromPage` and `nToPage` fields; otherwise, all pages are selected and we use the `nMinPage` and `nMaxPage` fields to set the first and last page to be printed:

[PRE56]

If the `PD_COLLATE` flags is present, the user has chosen to sort the pages:

[PRE57]

Finally, we create and return a pointer to the `Graphics` object to be used when painting to the printer.

[PRE58]

If the user terminates the dialog by pressing the **Cancel** button, we return null:

[PRE59]

# Print preview

The `PrintPreviewDocument` class displays the pages of the document parent window. The `OnKeyDown` method closes the document when the user presses the ***Esc*** key. The `OnSize` method adjusts the physical size of the page so that the page always fits inside the window. The `OnVerticalScroll` method shifts the pages when the user scrolls up or down, and `OnPaint` calls `OnPrint` of the parent document for each page.

**PrintPreviewDocument.h**:

[PRE60]

The `OnSize` function is overridden only to neutralize its functionality in `Document`. In `Document`, `OnSize` modifies the scroll bars, but we do not want that to happen in this class:

[PRE61]

The `page` field holds the current page number and `totalPages` holds the total number of pages:

[PRE62]

**PrintPreviewDocument.cpp**

[PRE63]

The constructor sets the `page` and `totalPages` fields to appropriate values.

[PRE64]

The horizontal scroll bar is always set to the width of the window, which means that the user cannot change its setting:

[PRE65]

The vertical scroll bar is set to match the number of pages of the document, and the scroll thumb corresponds to one page:

[PRE66]

The header displays the current and total number of pages:

[PRE67]

## Keyboard input

The `OnKeyDown` function is called when the user presses a key. If they press the ***Esc*** key, the preview window is closed and destroyed, and the input focus is returned to the main window of the application. If they press the ***Home*** , ***End*** , ***Page Up*** , or ***Page Down*** keys or the up and down arrow keys, `OnVerticalScroll` is called to take the appropriate action:

[PRE68]

We return `true` to indicate that the keyboard input has been used:

[PRE69]

## Scroll bar

The `OnVerticalScroll` function is called when the user scrolls the vertical bar. If they click on the scroll bar itself, above or below the scroll thumb, the previous or next page is displayed. And if they drag the thumb to a new position, the corresponding page is calculated. The `SB_TOP` and `SB_BOTTOM` cases are included to accommodate the ***Home*** and ***End*** keys from the preceding `OnKeyDown` function rather than to accommodate any scroll movements; they set the page to the first or last page:

[PRE70]

If the scroll movement has resulted in a new page, we set the header and the scroll bar position and invalidate and update the window:

[PRE71]

The `OnPaint` function in `PrintPreviewDocument` calls `OnPaint` in the parent standard document window in order to paint the contents of the preview window:

[PRE72]

# Summary

In this chapter, we looked into the registry, the clipboard, standard dialogs, and print preview. In [Chapter 14](ch14.html "Chapter 14. Dialogs, Controls, and Page Setup"), *Dialogs, Controls, and Page Setup*, we will look into custom dialogs, controls, converters, and page setup.