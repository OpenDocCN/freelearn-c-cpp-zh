# Chapter 11. The Document

In the previous chapter we looked into the implementation of the `Application` and `Window` classes, which are useful for general Windows applications. In this chapter, we will look into the implementation of the `Document`, `StandardDocument`, `Menu`, and `Accelerator` classes, which are useful for document-based Windows applications.

# The Document class

In this book, a **document** is a window intended for common document-based applications, such as the drawing program, spreadsheet program, and word processor of this book. The `Document` class implements the document described previously and is a direct subclass of the `Window` class. It supports caret and dirty flag, keyboard status, menus, accelerators, the mouse wheel, scroll bars, and drop files.

**Document.h**

[PRE0]

The keyboard holds either the `insert` or `overwrite` mode.

[PRE1]

Similar to `Window`, `Document` has a public constructor intended for instantiation and a protected constructor intended for subclasses. A document of the `Document` class can accept drop files, and the line size is used by the scroll bar methods:

[PRE2]

A dirty flag is set if the window has been modified and needs to be saved before closing (the document has been *dirty*). The content of the document can be zoomed in accordance with a zoom factor; the default is 1.0\. The name of the document is displayed in the document header by `GenerateHeader`, together with the zoom factor expressed as a percentage, and an asterisk (*****) if the dirty flag is `true`. However, the zoom factor is not displayed if it is 100%:

[PRE3]

The `OnSize` method is overridden to modify the size of the scroll bar in accordance with the client size. Note that the parameter to `OnSize` is the logical size of the client area, not the size of the window:

[PRE4]

The `OnMouseWheel` method is overridden to scroll the vertical scroll bar one line for each wheel click:

[PRE5]

The `Document` class supports the caret, and the `OnGainFocus` and `OnLoseFocus` methods are overridden to show or hide the caret. The `SetCaret` and `ClearCaret` methods create and destroy the caret:

[PRE6]

The `UpdateCaret` method is called when the caret needs to be modified, it is intended to be overridden and its default behavior is to do nothing:

[PRE7]

The `SetMenuBar` method sets the menu bar of the window. The `OnCommand` method is called every time the user selects a menu item or presses an accelerator key, and `CommandInit` is called before the menus become visible in order to set a check mark or a radio button at the menu item or to enable or disable it:

[PRE8]

If the `acceptDropFiles` parameter in the constructor is `true`, the document accepts drop files. If the user moves one or several files and drops them in the document window, `OnDropFile` is called with the list of path names as parameters. It is intended to be overridden by subclasses, and its default behavior is to do nothing:

[PRE9]

The `GetKeyboardMode` and `SetKeyboardMode` methods set and get the `keyboard` mode. The `OnKeyboardMode` method is called when the `keyboard` mode is changed; it is intended to be overridden and its default behavior is to do nothing:

[PRE10]

The `OnHorizontalScroll` and `OnVerticalScroll` methods handle the scroll messages. The scroll bar is set in accordance with the message settings:

[PRE11]

The `KeyToScroll` method takes a key and performs an appropriate scroll bar action depending on the key and whether the ***Shift*** or ***Ctrl*** key is pressed. For instance, the ***Page Up*** key moves the vertical scroll bar one page upward:

[PRE12]

The following methods set or get the logical position, line size, page size, and total size of the horizontal and vertical scroll bar:

[PRE13]

The command map stores the menu items of the document; for each menu item, the selection, enable, check, and radio listeners are stored:

[PRE14]

The accelerator set holds the accelerators of the document irrespective of whether it is a regular key or virtual key (for instance, ***F2*** , ***Home*** , or ***Delete*** ) and whether the ***Ctrl*** , ***Shift*** , or ***Alt*** key is pressed. The set is used by the message loop in `Application`:

[PRE15]

The `name` field is the name of the document displayed at the top of the window; `caretPresent` is true when the caret is visible:

[PRE16]

When the user presses one of the arrow keys, `OnKeyDown` is called. However, if `OnKeyDown` returns `false`, the scroll bar is changed; in that case, we need `lineSize` to define the size of a line to be scrolled:

[PRE17]

The `dirtyFlag` field is `true` when the user has changed the document without saving, resulting in the **Save** menu item being enabled and the user being asked whether to save the document when closing the window or exiting the application:

[PRE18]

The `menuBarHandle` method is the Win32 API function that handles the menu bar of the document window:

[PRE19]

The keyboard can hold the `insert` or `overwrite` mode, which is stored in `keyboardMode`:

[PRE20]

The `DocumentProc` method is called when the document window receives a message, similar to `WindowProc` in the `Window` class:

[PRE21]

The `ExtractPathList` method extracts the paths of the dropped files when the window receives the `WM_DROPFILES` message:

[PRE22]

## Initialization

The first `Document` constructor takes the coordinate system, the page size, parent window, style, appearance, whether the document accepts drop files, and the line size as its parameters. The size of a US Letter page in portrait mode (standing up) is 215.9 * 279.4 millimeters. A line (used by `KeyToScroll` when scrolling lines) is 5 millimeters in both the horizontal and vertical directions. Since a logical unit is one hundredth of a millimeter, we multiply each measure by one hundred.

**Document.cpp**

[PRE23]

The first constructor calls the second constructor with the `Windows` class named `Document` as the first parameter:

[PRE24]

The second constructor takes the same parameters as the first construct with the exception that it inserts the `Windows` class name as its first parameter:

[PRE25]

The range and page size of the scroll bars are stored in the window's scroll bar settings. However, the size of the line needs to be stored in `lineSize`:

[PRE26]

The header appears on the top bar of the document window:

[PRE27]

The default position of the scroll bars is `0`:

[PRE28]

The size of the scroll bars is the logical width and height of the page:

[PRE29]

The page sizes of the scroll bars represent the visible part of the document, which is the logical size of the client area:

[PRE30]

The Win32 API function `DragAcceptFiles` makes the window accept drop files. Note that we need to convert the C++ `bool` type of `acceptDropFiles` to the value `TRUE` or `FALSE` of the Win32 API `BOOL` type:

[PRE31]

The destructor destroys the caret if present:

[PRE32]

## The Document header

The `GetName` method simply returns the name. However, `SetName` sets the name and regenerates the header of the document window. The same goes for `SetZoom` and `SetDirty`: they set the zoom factor and dirty flag and then regenerate the header:

[PRE33]

The title of the document includes its name, whether the dirty flag is set (indicated by an asterisk), and the zoom status (as a percentage), unless it is 100%.

[PRE34]

`OnSize` modifies the page sizes of the horizontal and vertical scroll bars in accordance with the new client size:

[PRE35]

## The caret

As mentioned in [Chapter 1](ch01.html "Chapter 1. Introduction"), *Introduction*, a caret is the marker indicating where to input the next character. It is a thin vertical bar in the `insert` mode and a block in the `overwrite` mode. The `OnGainFocus` and `OnLoseFocus` methods show and hide the caret, if present:

[PRE36]

The `SetCaret` method displays a caret with the given dimensions. If there already is a caret present, it is destroyed:

[PRE37]

The size of the caret must be given in device units; there is a risk that the `LogicalToDevice` call rounds the width to zero (in the case of a vertical bar), in which case the width is set to 1:

[PRE38]

The new caret is created by the Win32 API functions `CreateCaret`, `SetCaretPos`, and `ShowCaret`:

[PRE39]

The `ClearCaret` method destroys the caret, if present:

[PRE40]

## The mouse wheel

When the user moves the mouse wheel, the vertical scroll bar is moved one line up or down (if they do not press the *Ctrl* key):

[PRE41]

If the user presses the ***Ctrl*** key, then the client area is zoomed. The permitted range is 10% to 1,000%:

[PRE42]

As the vertical scroll bar position has been modified, we need to repaint the whole client area:

[PRE43]

## The menu bar

The menu bar of the document is set by calling the Win32 API function `SetMenu`, which handles the document window and the menu bar; `menuBarHandle` is used when enabling or marking menu items in `OnCommandInit`, as shown here:

[PRE44]

The `OnCommand` method is called when the user selects a menu item or an accelerator. It looks up and calls the selection listener associated with the given command identity number:

[PRE45]

The `OnCommandInit` method is called before a menu becomes visible. It iterates through every menu item and, for each of them, decides whether it should be annotated with a check mark or radio button, or enabled or disabled:

[PRE46]

If the enable listener is not null, we call it and set the enable flag to `MF_ENABLED` or `MF_GRAYED` (disabled):

[PRE47]

If the check or radio listeners are not null, we call them and set `checkflag` or `radioFlag`:

[PRE48]

If either `checkFlag` or `radioFlag` is `true`, we check the menu item. Whether the menu item thereby becomes annotated with a check mark or a radio button is decided when the menu item is added to the menu, which is described in the `Menu` class in the next section. It is also stated in `Menu` that at least one of the check mark and radio listeners must be null, since it is not possible to annotate a menu item with both a check mark and a radio button:

[PRE49]

## The scroll bar

The `OnHorizontalScroll` and `OnVerticalScroll` methods are called every time the user scrolls by clicking the scroll bar arrows, the scroll bar itself, or by dragging the scroll thumb.

The `scrollPos` field holds the current scroll bar setting. The `scrollLine` variable is the size of the line, `scrollPage` is the size of the page (representing the logical size of the visible part of the document and equal to the logical size of the client area), and `scrollSize` is the total size of the scroll bar (representing the logical size of the document):

[PRE50]

In the case of leftward movement, we need to verify that the new scroll position doesn't go below zero:

[PRE51]

In the case of rightward movement, we need to verify that the scroll position does not exceed the scroll bar size:

[PRE52]

If the user drags the scroll bar thumb, we just set the new scroll position. The difference between the messages is that `SB_THUMBTRACK` is sent continually as the user drags the thumb, while `SB_THUMBPOSITION` is sent when the user releases the mouse button:

[PRE53]

Vertical scroll bar movements work in the same way as horizontal scroll bar movements:

[PRE54]

The `KeyToScroll` function is called when the user presses a key. It examines the key, performs an appropriate scroll action, and returns `true` if the key was used, indicating as much:

[PRE55]

If the scroll position has been changed, we set the new scroll position by calling the Win32 API function `SetScrollPos` and update the window and the caret:

[PRE56]

The Win32 API function `GetScrollPos` returns the current scroll bar position:

[PRE57]

The methods for the vertical scroll position work in the same way as the methods for the horizontal scroll bar:

[PRE58]

The `SetHorizontalScrollLineWidth`, `GetHorizontalScrollLineHeight`, `SetVerticalScrollLineHeight`, and `GetVerticalScrollLineHeight` methods have no Win32 API counterparts. Instead, we store the size of a scrolled line in the `lineSize` field:

[PRE59]

The `SetHorizontalScrollPageWidth`, `GetHorizontalScrollPageWidth`, `SetVerticalScrollPageHeight`, and `GetVerticalScrollPageHeight` methods have no direct Win32 API counterparts. However, the `GetScrollInfo` and `SetScrollInfo` functions handle the general scroll information, and we can set and extract the page information:

[PRE60]

The `SetHorizontalScrollTotalWidth`, `GetHorizontalScrollTotalWidth`, `SetVerticalScrollTotalHeight`, and `GetVerticalScrollTotalHeight` methods call the Win32 API functions `SetScrollRange` and `GetScrollRange`, which set and get the minimum and maximum scroll values. However, we ignore the minimum value since it is always 0:

[PRE61]

## The DocumentProc method

The `DocumentProc` method is called every time the document (of the `Document` class) receives a message. If it uses the message, 0 is returned; otherwise, `WindowProc` (described in the previous chapter) is called to further process the message:

[PRE62]

We look up the window in `WindowMap` in the `Window` class and take action only if the window is a `Document` object:

[PRE63]

The direction of the mouse wheel is downward if the word parameter's ninth bit is set:

[PRE64]

The key-down messages both check the ***Insert*** key and call `OnKeyDown` and `KeyToScroll`, returning 0 if one of them uses the key:

[PRE65]

If the user presses the *Insert* key, the keyboard mode is swapped between the insert and overwrite mode. `SetKeyboardMode` sets the keyboard mode and calls `OnKeyboardMode`, which is intended to be overridden by subclasses to alert the application of the change:

[PRE66]

If the user does not press the ***Insert*** key, we check whether `OnKeyDown` uses the key (and thereby returns `true`). If it does not, we instead check whether `KeyToScroll` uses the key. If either `OnKeyDown` or `KeyToScroll` returns `true`, 0 is returned:

[PRE67]

The `WM_COMMAND` case is sent when the user selects a menu item, and `WM_INITMENUPOPUP` is sent before a menu becomes visible. Messages are handled by calling `OnCommand`, which executes the selection listener connected to the menu item, and `OnCommandInit`, which enables or annotates menu items with check marks or radio buttons before they become visible:

[PRE68]

When the user drops a set of files into the window, we need to extract their paths before calling `OnDropFile`. The `ExtractPath` method extracts the path of the files from the drop and returns a list of paths, which is sent to `OnDropFile`:

[PRE69]

The `WM_HSCROLL` and `WM_VSCROLL` messages are handled by calling their matching methods:

[PRE70]

Finally, if the message is not caught by `DocumentProc`, `WindowProc` (from the previous chapter) is called to further process the message:

[PRE71]

The `ExtractPathList` method extracts the paths of the dropped files by calling the Win32 API function `DragQueryFile` and returns the list of paths:

[PRE72]

The `DragQueryFile` method returns the number of files when the second parameter is `0xFFFFFFFF`:

[PRE73]

The `DragQueryFile` method returns the size of the path string when the second parameter is a zero-based index and the third parameter is null:

[PRE74]

The `DragQueryFile` method copies the path itself when the third parameter is a pointer to a text buffer rather than null:

[PRE75]

# The Menu class

The `Menu` class handles a menu, made up of a list of menu items, separator bars, or submenus. When a menu item is added, its command information is stored in the document's command map to be used when receiving the `WM_COMMAND` and `WM_INITCOMMAND` messages. If the menu item text includes an accelerator, it is added to the document's accelerator set. The `Command` class is an auxiliary class holding pointers to the menu items: selection, enable, check, and radio listeners.

**Command.h**

[PRE76]

**Command.cpp**

[PRE77]

Menu and accelerator listeners are not regular methods. They are declared (they do not need to be defined) by the `DECLARE_BOOL_LISTENER` and `DECLARE_VOID_LISTENER` macros. This is because we cannot call a non-static method in an unknown class directly. Therefore, we let the macros declare a non-static method without parameters and define a static method with a `void` pointer as a parameter that calls the non-static method. The macros do not define the non-static method. That task is left for the user of Small Windows.

When the user adds a menu item with a listener, a `Command` object is created. It is actually the static method with the `void` pointer parameter that is added to the `Command` object. Moreover, when the user selects a menu item, it is the static method that is called. The static method in turn calls the non-static method, which is defined by the user.

The macros take the names of the current class and the listener as parameters. Note that the `bool` listener is constant, while the `void` listener is not constant. This is because `bool` listeners are intended to look up the values of one or several of the fields of the class, while `void` listeners also modify the fields.

**Menu.h**

[PRE78]

The document pointer is needed when accessing the command map and accelerator set of the document. Every menu except the menu bar has text that is displayed in the document window; `menuHandle` is the Win32 API menu handle wrapped by this class:

[PRE79]

**Menu.cpp**

[PRE80]

The constructor initializes the pointer document and the text. It also creates the menu by calling the Win32 API function `CreateMenu`. Since the menu bar does not need text, the `text` parameter is empty by default:

[PRE81]

The copy constructor copies the fields of the menu. Note that we copy the `menuHandle` field rather than creating a new menu handle.

[PRE82]

The `AddMenu` method adds a menu (not a menu item) as a submenu to the menu, while `AddSeparator` adds a separator (a horizontal bar) to the menu:

[PRE83]

The `AddItem` method adds a menu item (not a menu) to the menu, with the selection, enable, check, and radio listeners:

[PRE84]

The selection listener is not allowed to be null, and at least one of the check marks and radio listeners must be null, since it is not possible to annotate a menu item with both a check mark and a radio button:

[PRE85]

Each menu item is given a unique identity number, which we obtain from the current size of the command map:

[PRE86]

We add a `Command` object to the command map and add the menu item with the Win32 API function `AppendMenu`, which takes the menu handle, identity number, and text:

[PRE87]

If the radio listener is not null, we need to call the Win32 API function `SetMenuItemInfo` in order for the radio button to appear with the menu item:

[PRE88]

Finally, we call `TextToAccelerator` in `Accelerator` (described in the next section) to add an accelerator, if present, to the accelerator set of the document, which is used by the message loop of `Application`:

[PRE89]

# The Accelerator class

It is possible to add an accelerator to a menu item. The accelerator text is preceded by a tabulator character (`\t`) and the text is made up of the optional prefixes `Ctrl+`, `Shift+`, or `Alt+` followed by a character (for instance, `&Open\tCtrl+O`) or the name of a virtual key (for instance, `&Save\tAlt+F2`).

**Accelerator.h**

[PRE90]

The Win32 API holds a set of virtual keys with names beginning with `VK_`. In Small Windows, they have been given other names, hopefully easier to understand. The virtual keys available are: **F1** - **F12**, **Insert**, **Delete**, **Backspace**, **Tab**, **Home**, **End**, **Page Up**, **Page Down**, **Left**, **Right**, **Up**, **Down**, **Space**, **Escape**, and **Return**:

[PRE91]

The `Accelerator` class only holds the `TextToAccelerator` method, which takes text, extracts the accelerator, and adds it to the accelerator set, if present:

[PRE92]

**Accelerator.cpp**

[PRE93]

`TextToVirtualKey` is an auxiliary function that takes text and returns the corresponding virtual key. The `keyTable` array holds the map between the texts and the available virtual keys:

[PRE94]

We loop through the table until we find the virtual key:

[PRE95]

If we do not find a key matching the text, an assert occurs:

[PRE96]

In `TextToAccelerator`, we store the **Control**, **Shift**, **Alt**, and virtual key status together with the key in a Win32 API `ACCEL` structure:

[PRE97]

First, we check whether the text contains a *Tab* key (**\t**). If it does, we initialize the `ACCEL` structure with `itemId` and extract the accelerator part of the text:

[PRE98]

If the accelerator text contains the prefix `Ctrl+`, `Alt+`, or `Shift+`, we mask `FCONTROL`, `FALT`, or `FSHIFT` to the `fVirt` field and remove the prefix:

[PRE99]

After we remove the `Ctrl+`, `Shift+`, and `Alt+` prefixes, we look into the remaining part of the accelerator text. If there is one single character (the length is one), we save it in the `key` field. However, we do not save the ASCII number. Instead, we save the letter number, which starts with 1 for `a` or `A`:

[PRE100]

If the remaining part of the accelerator text is made up of more than one character, we assume that it is a virtual key and call `TextToVirtualKey` to find it and mask the `FVIRTKEY` constant to the `fVirt` field:

[PRE101]

If `fVirt` is still zero, the accelerator does not contain `Ctrl+`, `Shift+`, `Alt+`, or a virtual key, which is not allowed:

[PRE102]

Finally, we add the accelerator to the accelerator set:

[PRE103]

Note that no accelerator is added to the accelerator set if the text does not contain a tabulator:

[PRE104]

# The StandardDocument class

The `StandardDocument` class is a direct subclass of `Document`; it handles the **File**, **Edit**, and **Help** menus and implements file handling, cut, copy, and paste, drop files, and printing. There is no specific message function for this class; all messages are sent to `DocumentProc` in the `Document` section covered previously. The document name and the dirty flag are automatically updated by the framework. `StandardDocument` does also handle the Page Setup dialog, which is more closely described in [Chapter 12](ch12.html "Chapter 12. The Auxiliary Classes"), *The Auxiliary Classes*.

**StandardDocument.h**

[PRE105]

Most constructor parameters are sent to the `Document` constructor. What is specific for `StandardDocument` is the file description text and the copy and paste format lists. The file description is used by the standard save and open dialogs. The copy and paste lists are used when copying and pasting information between the application and the global Clipboard:

[PRE106]

The `StandardFileMenu`, `StandardEditMenu`, and `StandardHelpMenu` methods create and return the standard menus. If `print` in `StandardFileMenu` is `true`, the **Page Setup**, **Print**, and **Print Preview** menu items are included:

[PRE107]

The **Save** menu item is disabled when the document does not need to be saved (the dirty flag is `false`). The `SaveEnable` method is called before the **Save** menu item becomes visible and enables it if the dirty flag is `true`.

[PRE108]

The `OnSave` method calls `SaveFileWithName` or `SaveFileWidhoutName` depending on whether the document has been given a name. However, `OnSaveAs` always calls `SaveFileWithoutName`, regardless of whether the document has a name.

[PRE109]

The `ClearDocument`, `WriteDocumentToStream`, and `ReadDocumentFromStream` methods are called when the user selects the **New**, **Save**, **Save As**, or **Open** menu items and are intended to be overridden by subclasses to clear, write, and read the document:

[PRE110]

The `OnCut`, `OnCopy`, `OnPaste`, and `OnDelete` methods are called when the user selects the corresponding menu item in the **Edit** menu. The default behavior for `OnCut` is to call `OnCopy` followed by `OnDelete`:

[PRE111]

The `CutEnable`, `CopyEnable`, `PasteEnable`, and `DeleteEnable` methods are listeners deciding whether the menu items are enabled. The default behavior for `CutEnable` and `DeleteEnable` is to call `CopyEnable`:

[PRE112]

The `IsCopyAsciiReady`, `IsCopyUnicodeReady`, and `IsCopyGenericReady` methods are called by `CopyEnable`. They are intended to be overridden and return `true` if the application is ready to be copied in the ASCII, Unicode, or generic formats. Their default behavior is to return `false`:

[PRE113]

The `CopyAscii`, `CopyUnicode`, and `CopyGeneric` methods are called by `OnCopy` when the user selects the **Copy** menu item. They are intended to be overridden by subclasses and are called in accordance with the copy format list in the constructor and the copy-ready methods:

[PRE114]

The `IsPasteAsciiReady`, `IsPasteUnicodeReady`, and `IsPasteGenericReady` methods are called by `PasteEnable`, which returns `true` if at least one of the methods returns `true`. They are intended to be overridden and return `true` if the application is ready to be pasted in the ASCII, Unicode, or generic formats. Their default behavior is to return `true`:

[PRE115]

The `PasteAscii`, `PasteUnicode`, and `PasteGeneric` methods are called by `OnPaste` when the user selects the **Paste** menu item. They are intended to be overridden by subclasses and are called in accordance with the paste format list in the constructor and the paste-ready methods. One difference between copying and pasting is that copying is performed in all available formats while pasting is performed in the first available format only:

[PRE116]

The `OnDropFile` methods is called when the user drops a set of files in the window's client area. If there is exactly one file with the suffix given in the constructor in the path list, that file is read in the same way as if the user had selected it in the standard open dialog. However, if there are no files or more than one file with the suffix in the list, an error message is displayed:

[PRE117]

The `PageOuterSize` methods returns the logical size of the page in portrait or landscape mode depending on the page setup settings, without regard to the margins, while `PageInnerSize`, `PageInnerWidth`, and `PageInnerHeight` return the size of the page after subtracting the margins:

[PRE118]

The `OnPageSetup`, `OnPrintPreview`, and `OnPrintItem` methods are called when the user selects the **Page Setup**, **Print**, and **Print Preview** menu items. They display **Page Setup Dialog**, **Print Preview Window**, and **Print Dialog**:

[PRE119]

The `PrintPage` method is called by `OnPrintItem` and prints one page of the document:

[PRE120]

The `OnPageSetup` method is called to notify the application when the user has selected the **Page Setup** menu item and has changed the page setup information. It is intended to be overridden by subclasses and its default behavior is to do nothing:

[PRE121]

The `GetTotalPages` method returns the number of pages to print; the default is 1\. It is intended to be overridden by subclasses:

[PRE122]

The `OnPrint` method is called once by `OnPrintItem` for each page and copy. Its default behavior is to write the header and footer in accordance with the setting in the **Page Setup Dialog**, and then call `OnDraw` for the application-specific contents of the document:

[PRE123]

The `OnExit` method is called when the user selects the **Exit** menu item and quits the application if `TryClose` returns `true`. If the dirty flag is `true`, `TryClose` displays a message box, asking the user for permission to close the window:

[PRE124]

The `OnAbout` method displays a simple message box with the application name:

[PRE125]

The `fileFilter` fields are used by the **Open** and **Save** standard dialogs and `fileSuffixList` is used to check the file suffix of dropped files:

[PRE126]

The `pageSetupInfo` field is used when the user selects the **Page Setup** menu item. It stores information about the header and footer text and font, page orientation (portrait or landscape), margins, and whether the pages are surrounded by a frame. Refer to the next chapter for a closer description.

[PRE127]

The `copyFormatList` and `pasteFormatList` fields hold the formats available for cutting, copying, and pasting:

[PRE128]

## Initialization

The first `StandardDocument` constructor takes a large set of parameters. The coordinate system, page size, parent window, style, appearance, whether the document accepts drop files, and the line size parameters are the same as in the `Document` case covered previously.

What remains is the file description text, whether the print menu is present, and the format list for copying and pasting. The description text holds a semicolon-separated list of file descriptions and file suffixes for the allowed files, for instance, **Calc Files**, *clc*; **Text Files**, *txt*. The copy and paste format list holds the allowed formats for copying and pasting information.

**StandardDocument.cpp**

[PRE129]

Most constructor parameters are sent to the `Document` constructor. However, the copy and paste format lists are stored in `copyFormatList` and `pasteFormatList`. The file filter and file suffix lists are initialized by `InitializeFileFilter`:

[PRE130]

In `Window`, we used the page size for transforming between logical and physical units. In `Document`, we used it for setting the scroll page size. However, in `StandardDocument`, there are actually two kinds of page sizes: the outer and inner page size. The outer page size is the page size without taking the margins of the document into consideration. The inner page size is obtained by subtracting the margins from the outer page size. In `StandardDocument`, we use the inner page size to set the size of the scroll bar:

[PRE131]

## Standard menus

The code for this is shown as follows:

[PRE132]

The standard **File** menu holds the **New**, **Open**, **Save**, **Save As**, and **Exit** menu items as well as (if `print` is `true`) the **Page Setup**, **Print Preview**, and **Print** menu items:

[PRE133]

The standard **Edit** menu holds the **Cut**, **Copy**, **Paste**, and **Delete** menu items:

[PRE134]

The standard **Help** menu holds the **About** menu item with the help of the application name:

[PRE135]

## File management

The `TryClose` method checks whether the dirty flag is `true` when the user tries to close the window. If it is `true`, the user is asked if they want to save the document before closing it. If they answer yes, the document is saved as if the user has selected the **Save** menu item. If the dirty flag is set to `false` after that, it means that the save operation went well and `true` is returned. If the user answers no, `true` is returned and the window is closed without saving. If the answer is cancel, `false` is returned and the closing is aborted:

[PRE136]

The `OnExit` method calls `TryClose` and deletes the application's main window, which eventually sends a quit message to the message loop that terminates the application, if `TryClose` returns `true`:

[PRE137]

The `OnNew` method is called when the user selects the **New** menu item. It tries to close the window by calling `TryClose`. If `TryClose` returns `true`, the document, dirty flag, and name are cleared, and the window is invalidated and updated. The `ClearDocument` method is indented to be overridden by subclasses to clear the application-specific contents of the document:

[PRE138]

The `OnOpen` method is called when the user selects the **Open** menu item. It tries to close the window by calling `TryClose` and displays the standard open dialog to establish the path of the file if it succeeds. If `OpenDialog` returns `true` and the input stream is valid, the page setup information is read and the methods `ClearDocument` and `ReadDocumentFromStream`, which are intended to be overridden by subclasses, are called:

[PRE139]

The **Save** menu item is enabled if the dirty flag is `true`:

[PRE140]

When saving the file, we call `SaveFileWithName` if the file has a name. If the file has not yet been given a name, `SaveFileWithoutName` is called instead:

[PRE141]

When the user selects **Save As**, `SaveFileWithoutName` is called and the **Save** standard dialog is displayed, regardless of whether the document has a name:

[PRE142]

The `SaveFileWithoutName` method displays the save dialog. If the user presses the **Ok** button, the `SaveDialog` call returns `true`, the new name is set, and `SaveFileWithName` is called to do the actual writing of the document file:

[PRE143]

The `SaveFileWithName` method tries to open the document file for writing and calls `WriteDocumentToStream`, which is intended to be overridden by subclasses, to do the actually writing of the document's content. If the writing of both the page setup information and the contents of the document succeeds, the dirty flag is cleared:

[PRE144]

When the user selects the **About** menu item in the **Help** standard menu, a message box with a message including the name of the application is displayed:

[PRE145]

## Cut, copy, and paste

The default behavior for `CutEnable` and `DeleteEnable` is to simply call `CopyEnable`, since it is likely that they are enabled under the same conditions:

[PRE146]

The default behavior for `OnCut` is to simply call `OnCopy` and `OnDelete`, which is the common action for cutting:

[PRE147]

The `OnDelete` method is empty and intended to be overridden by subclasses:

[PRE148]

The `CopyEnable` method iterates through the paste format list and calls `IsCopyAsciiReady`, `IsCopyUnicodeReady`, or `IsCopyGenericReady` depending on the formats. As soon as one of the methods returns `true`, `CopyEnable` returns `true`, implying that it is enough that copying is allowed for one of the formats. When the actual copying occurs in `OnCopy`, the ready methods are called again:

[PRE149]

The `OnCopy` method iterates through the copy format list given in the constructor and calls appropriate methods depending on the formats:

[PRE150]

If the ASCII format applies and if `IsCopyAsciiReady` returns `true`, `CopyAscii` is called, which is intended to be overridden by subclasses to fill `asciiList` with ASCII text. When the list has been copied, it is passed on to `WriteAscii` in `Clipboard`, which stores the text on the global clipboard:

[PRE151]

If the Unicode format applies and if `IsCopyUnicodeReady` returns `true`, `CopyUnicode` is called, which is intended to be overridden by subclasses to fill `unicodeList` with Unicode text. When the list has been copied, it is passed on to `WriteUnicode` in `Clipboard`, which stores the text on the global clipboard:

[PRE152]

If neither ASCII nor Unicode applies and if `IsCopyGenericReady` returns `true`, `CopyGeneric` is called, which is intended to be overridden by subclasses to fill the character list with generic information. In C++, a value of type `char` always holds one byte; it is therefore used in the absence of a more generic byte type. When the information has been copied to `infoList`, it is passed on to `WriteGeneric` in `Clipboard` to store the information on the global Clipboard:

[PRE153]

The `PasteEnable` method iterates through the paste format list given in the constructor and returns `true` if at least one of the formats is available on the global Clipboard:

[PRE154]

The `OnPaste` method iterates through the paste format list given in the constructor and, for each format, checks whether it is available on the global Clipboard. If it is, an appropriate method is called. Note that, while `OnCopy` iterates through the whole copy format list, `OnPaste` quits after the first format available on the Clipboard, which makes the order of the paste format list significant:

[PRE155]

In the case of the ASCII format, `ReadAscii` in `Clipboard` is called, which reads the text list from the global clipboard and, if `IsPasteAsciiReady` returns `true`, calls `PasteAscii`, which is intended to be overridden by subclasses to do the actual application-specific pasting:

[PRE156]

In the case of the Unicode format, `ReadUnicode` in `Clipboard` is called, which reads the text list from the global clipboard and, if `IsPasteUnicodeReady` returns `true`, it calls `PasteUnicode`, which is intended to be overridden by subclasses to do the actual application-specific pasting:

[PRE157]

If neither ASCII nor Unicode applies, `ReadGeneric` in `Clipboard` is called to read the generic information from the global clipboard and, if `IsPasteGenericReady` returns `true`, it calls `PasteGeneric`, which is intended be overridden by subclasses to do the actual pasting.

One difference between copying and pasting in the generic case is that `OnCopy` uses a character list since it does not know the size in advance (if we used a memory block, we would need two methods: one that calculates the size of the block and one that does the actual reading, which would be cumbersome), while `OnPaste` uses a memory block, which cannot be converted into a character list since we do not know the size. Only the document-specific overridden version of `PasteGeneric` can decide the size of the memory block:

[PRE158]

## Drop files

When the user drops one or several files in the client area of the window, we check the file suffix of each filename. If we find exactly one file with one of the file suffixes of the document (the `fileSuffixList` field) we open it in the same way as if the user had opened it with the standard **Open** dialog:

[PRE159]

We iterate through the path list and add every path with the file suffix to `pathSet`:

[PRE160]

If `pathSet` is empty, no files with the file suffix have been dropped.

[PRE161]

If `pathSet` holds more than one file, too many files with the file suffix have been dropped:

[PRE162]

If `pathSet` holds exactly one file, it is read in the same way as if the user has selected the **Open** menu item:

[PRE163]

## Page size

The `PageOuterSize` method returns the page size with no regard to the margins. There are two page sizes, depending on the orientation in the **Page Setup** dialog. The page size given in the constructor refers to the `Portrait` orientation. In the case of the `Landscape` orientation, the width and height of the page are swapped:

[PRE164]

The `PageInnerSize` method returns the page size with regard to the margins. The width is subtracted by the left and right margins. The height is subtracted by the top and bottom margins. Remember that the margins are given in millimeters and the logical units are in hundredths of millimeters. Therefore, we multiply the margins by 100:

[PRE165]

The `PageInnerWidth` and `PageInnerHeight` methods return the width and height of the document after the margins have been subtracted. As the margins are given in millimeters and one millimeter is one hundred logical units, we multiply the margins by 100 in order to obtain logical units:

[PRE166]

## Page setup

The `OnPageSetup` method is called when the user selects the **Page Setup** menu item. It displays the **Page Setup** dialog (refer to [Chapter 12](ch12.html "Chapter 12. The Auxiliary Classes"), *The Auxiliary Classes*) and calls `OnPageSetup`, which is intended to be overridden by subclasses, to notify the application that the page setup information has been changed:

[PRE167]

## Printing

The `OnPrintPreview` method is called when the user selects the **Print Preview** menu item. It displays the print preview document, which is more closely described in [Chapter 12](ch12.html "Chapter 12. The Auxiliary Classes"), *The Auxiliary Classes*. The `GetTotalPages` method returns the current number of pages in the document:

[PRE168]

The `OnPrintItem` method is called when the user selects the **Print** menu item. It displays the standard **Print** dialog and prints the pages of the document in accordance with the page interval and the order and number of copies specified by the user in the dialog.

The method is named `OnPrintItem` so that it is not confused with `OnPrint` in `Window`, which is called when the window receives the `WM_PAINT` message. However, both methods could have been named `OnPrint` since they have different parameter lists:

[PRE169]

The `PrintDialog` method creates and returns a pointer to a `Graphics` object, if the user presses the **Ok** button, or a null pointer if the user presses the **Cancel** button. The `totalPages` parameters indicate the last possible page that the user can choose (the first possible page is 1). In the case of the **Ok** button, `firstPage`, `lastPage`, `copies`, and `sorted` are initialized: `firstPage` and `lastPage` are the page intervals to be printed, `copies` is the number of copies to be printed, and `sorted` indicates whether the copies (if more than one) will be sorted:

[PRE170]

The Win32 API function `StartDoc` initializes the printing process. It takes the device context connected to the printer by the `Graphics` object and a `DOCINFO` structure that only needs to be initialized with the document name. If `StartDoc` returns a value greater than zero, we are clear to print the pages. We prepare the device context and disable the window while the printing occurs:

[PRE171]

If `sorted` is `true`, the pages are printed in the sorted order. For instance, let's assume that `firstPage` is set to 1, `lastPage` is set to 3, and `copies` is set to 2\. If `sorted` is `true`, the pages are printed in order 1, 2, 3, 1, 2, 3\. If `sorted` is `false`, they are printed in the order 1, 1, 2, 2, 3, 3\. `PrintPage` is called for each page and the printing continues as long as it returns true; `printOk` keeps track of whether the loop continues:

[PRE172]

The Win32 API function `EndDoc` is used to finish printing:

[PRE173]

The `PrintPage` method calls the Win32 API functions `StartPage` and `EndPage` before and after the printing of the page. If they both return values greater than zero, it indicates that the printing went well, `true` is returned, and more pages can be printed. `OnPrint` (overridden from `Window`) is called to do the actual printing, `page` and `copy` are the current page and copy, and `totalPages` is the number of pages in the document:

[PRE174]

The `OnPrint` method prints the information given by the `pageSetupInfo` field. Then, the contents of the documents are clipped and drawn by calling `OnDraw`, and finally the frame enclosing the contents of the document is drawn, if present:

[PRE175]

The document is cleared by being painted white.

[PRE176]

The header text is written unless it is empty; if the current page is the first page, it is not written:

[PRE177]

Similar to the header text, the footer text is written unless it is empty; if the current page is the first page, it is not written:

[PRE178]

The current state of the device context is saved, the origin is set to the top-left corner of the current page, the area of the current page is clipped, `OnDraw` is called to draw the current page, and the paint area is finally restored:

[PRE179]

Finally, the page is enclosed by a rectangle if the frame field of the page setup information is `true`:

[PRE180]

# Summary

In this chapter, we studied the document classes of Small Windows: `Document`, `Menu`, `Accelerator`, and `StandardDocument`. In [Chapter 12](ch12.html "Chapter 12. The Auxiliary Classes"), *The Auxiliary Classes*, we continue by looking into to the auxiliary classes of Small Windows.