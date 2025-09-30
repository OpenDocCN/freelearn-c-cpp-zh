# Chapter 6. Building a Word Processor

In this chapter, we build a word processor that is capable of handling text on character level: that is, a single character that has its own font, color, size, and style. We also introduce caret handling, printing and print previewing, and file dropping, as well as clipboard handling with ASCII and Unicode text, which means that we can cut and paste between this application and, for instance, a text editor.

![Building a Word Processor](img/image_06_001.jpg)

# Auxiliary classes

A document in this application is made up of pages, paragraphs, lines, and characters. Let me try to explain how it all hangs together:

*   First of all, the document is made up of a list of characters. Each character has its own font and pointers to the paragraph and line it belongs to. The character information is stored in objects of the `CharInfo` class. The `charList` field in the `WordDocument` class is a list of `CharInfo` objects.
*   The characters are divided into paragraphs. A paragraph does not hold its own character list. Instead, it holds the indexes in the character list of its first and last characters. The `paragraphList` field in `WordDocument` is a list of `Paragraph` objects. The last character of each paragraph is always a newline.
*   Each paragraph is divided into a list of lines. The `Paragraph` class below holds a list of `Line` objects. A line holds the indexes of its first and last characters relative to the beginning of the paragraph.
*   Finally, the document is also divided into pages. A page holds as many whole paragraphs as possible.

Every time something is changed in the document, the current line and paragraph are recalculated. The page list is also recalculated.

Let's continue to look into the `CharInfo`, `LineInfo`, and `Paragraph` classes.

## Character information

The `CharInfo` class is a structure that holds the following:

*   A character and its font
*   Its enclosing rectangle, which is used when drawing the character
*   Pointers to the line and the paragraph it belongs to

**CharInfo.h**

[PRE0]

Each of the private fields in this class has its own method for getting and setting the value. The first set of methods is constant and returns the value itself, which means that the value of the field cannot be changed by these methods. The second set of methods is nonconstant and returns a reference to the field, which means that the value can be changed. However, they cannot be called from a constant object.

[PRE1]

The `tChar` and `charFont` fields hold the character itself and its font, and the `charRect` coordinates are relative to the top-left position of the paragraph the character belongs to. Each character belongs to a paragraph and one of the lines of that paragraph, which `paragraphPtr` and `lineInfoPtr` point at.

[PRE2]

**CharInfo.cpp**

[PRE3]

The default value of the `font` parameter is the system font that gives the default font. It is often 10 point Arial.

[PRE4]

The copy constructor and assignment operator copies the fields. They are called on several occasions when the characters are written to and read from file streams, or when they are cut, copied, or pasted.

[PRE5]

The `WriteCharInfoToStream` method writes and the `ReadCharInfoFromStream` method reads the values of the class to and from a file stream and the clipboard. Note that we omit the `paragraphPtr` and `lineInfoPtr` pointers since it would be meaningless to save pointer addresses to a stream. Instead, their values are set by the `ReadDocumentFromStream` method in the `WordDocument` class after calling the `ReadCharInfoFromStream` method.

[PRE6]

The `WriteCharInfoToClipboard` method writes and the `ReadCharInfoFromClipboard` method reads the values to and from the clipboard. Also, in this case, we omit the `paragraphPtr` and `lineInfoPtr` pointers. These pointers are set by the `PasteGeneric` method in the `WordDocument` class after the call to the `ReadCharInfoFromClipboard` method.

[PRE7]

## Line information

The `LineInfo` method is a small structure holding information about a line in a paragraph:

*   The integer index of its first and last characters
*   Its height and ascent, that is, the height and ascent of the largest character on the line.
*   The top position of the line relative to its paragraph top position

**LineInfo.h**

[PRE8]

Similar to the `CharInfo` method mentioned previously, the `LineInfo` method holds a set of constant methods for inspecting the class fields and a set of nonconstant methods for modifying them.

[PRE9]

The fields of this class are four integer values; the `first` and `last` fields refer to the first and last characters on the line, respectively. The `top`, `height`, and `ascent` fields are the top position of the line relative to the top position of the paragraph, the maximum height, and ascent of the line.

[PRE10]

**LineInfo.cpp**

[PRE11]

The default construct is called when the user reads a document from a stream, while the second constructor is called when new lines of a paragraph are being generated.

[PRE12]

The `WriteLineInfoToStream` and `ReadLineInfoFromStream` methods simply write and read, respectively, the field value. Note that there are no corresponding methods for cut, copy, and paste since the line list of a paragraph is regenerated each time the paragraph is pasted.

[PRE13]

## The Paragraph class

A document is made up of a sequence of paragraphs. The `Paragraph` structure holds the following:

*   The index of its first and last characters
*   Its top position relative to the beginning of the document, and its height
*   Its index in the document paragraph pointer list
*   Its alignment–a paragraph can be left, center, justified, or right aligned
*   Whether it holds a page break, that is, whether this paragraph will be located at the beginning of the next page

**Paragraph.h**

[PRE14]

As you can see, we name the `AlignmentField` method instead of just the `Alignment` method. The reason for this is that there already is a class named `Alignment`. We cannot give the same name to both the class and method. Therefore, we add the `Field` suffix to the method name.

[PRE15]

The `first` and `last` fields are the index in the document character list of the first and last characters in the paragraph, respectively; the last character of the paragraph is always a newline. The `top` field is the top position of the paragraph relative to the beginning of the document, which is always zero for the first paragraph of the document and positive for the other paragraphs. The `height` is the height of the paragraph, and `index` refers to the index of the paragraph in the document paragraph pointer list. If `pageBreak` is `true`, the paragraph will always be located at the beginning of a page.

[PRE16]

A paragraph can be left, right, centered, and justified aligned. In the justified case, the spaces are extended in order for the words to be distributed over the whole width of the page.

[PRE17]

A paragraph is made up of at least one line. The indexes of the `linePtrList` list are relative to the index of the first character in the paragraph (not the document), and the coordinates are relative to the top of the paragraph (again, not the document).

[PRE18]

**Paragraph.cpp**

[PRE19]

The idea is that the `WriteParagraphToStream` and `ReadParagraphFromStream` methods write and read, respectively, all information about the paragraph. Remember that all coordinates are given in logical units (hundredths of millimeters), which means that works to save and open the file on screens with different resolutions.

[PRE20]

When we have read indexes of the first and last character of the paragraph, we need to set the paragraph pointer of each character.

[PRE21]

In the same way as in the paragraph pointer case above, we need to set the line pointer of each character.

[PRE22]

On the other hand, the `WriteParagraphToClipboard` and `ReadParagraphFromClipboard` methods only write and read, respectively, the essential information. After the paragraph has been read, the `CalaulateParagraph` method is then called, which calculates the character rectangles and the height of the paragraph and generates its line pointer list.

[PRE23]

# The MainWindow class

The `MainWindow` class is nearly identical to the versions of the previous chapters. It sets the application name to `Word` and returns the address of a `WordDocument` instance:

[PRE24]

# The WordDocument class

The `WordDocument` class is the main class of the application. It extends the `StandardDocument` class and takes advantage of its document-based functionality.

**WordDocument.h**

[PRE25]

The `InitDocument` class is called by the constructor, the `ClearDocument`, and `Delete` classes.

[PRE26]

The `OnKeyboardMode` method is called every time the user presses the *Insert* key. The `UpdateCaret` method sets the caret to a vertical bar in `insert` mode and a block in `overwrite` mode. When the user marks one or several characters, the caret is cleared.

[PRE27]

When the user presses, moves, and releases the mouse, we need to find the index of the character located at the mouse position. The `MousePointToIndex` method finds the paragraph, and the `MousePointToParagraphIndex` method finds the character in the paragraph. The `InvalidateBlock` method invalidates the characters from the smallest index, inclusive, to the largest index, exclusive.

[PRE28]

When the user double-clicks on a word, it will be marked. The `GetFirstWordIndex` and `GetLastWordIndex` methods find the first and last index of the word, respectively, if in fact the user double-clicks on a word (rather than a space, period, comma, or question mark).

[PRE29]

In this application, we introduce touchscreen handling. Unlike mouse clicks, it is possible to touch the screen in several locations at the same time. Therefore, the parameter is a list of points rather that one individual point.

[PRE30]

The `OnPageSetup` method is called when the user has changed the page setting by selecting the **Page Setup** menu item in the **File** menu, which allows the user to modify the page and paragraphs settings. The `CalculateDocument` method distributes the paragraphs on the pages. If a paragraph is marked with a page break, or if it does not completely fit on the rest of the current page, it is placed at the beginning of the next page.

[PRE31]

Unlike the applications in the previous chapters, we override both the `OnPaint` and `OnDraw` methods. The `OnPaint` method is called when the client area needs to be redrawn. It performs paint-specific actions, that is, actions that will be performed only when the document is drawn in a window, but not when it is sent to the printer. More specifically, we add page break markers in the client area, but not in the printer text.

The `OnPaint` method then calls the `OnDraw` method that performs the actual drawing of the document. There is also a method `OnPrint` in the `StandardDocument` class (which we do not override) that calls the `OnDraw` method when printing the document.

[PRE32]

Similar to the applications in the previous chapters, the `ClearDocument`, `WriteDocumentToStream`, and `ReadDocumentFromStream` methods are called when the user selects the **New**, **Save**, **Save As**, or **Open** menu items in the **File** menu.

[PRE33]

The `CopyEnable` method returns `true` when text is ready to be copied, that is, when the user has marked a part of the text. The `CopyAscii` and `CopyUnicode` methods are called when the user selects the **Cut** or **Copy** menu item and copies the marked text into a string list. The `CopyGeneric` method is also called when the user selects the **Cut** or **Copy** menu item and copies the marked text in an application-specific format that also copies the font and style of the characters.

[PRE34]

The `PasteAscii`, `PasteUnicode`, and `PasteGeneric` methods are called when the user selects the **Paste** menu item. One difference between copying and pasting is that all the three aforementioned methods are called when copying, but only one method when pasting, in the order the format is given in the `StandardDocument` constructor call.

[PRE35]

We do not override the `CutEnable` or `OnCut` methods, since the `CutEnable` method in the `StandardDocument` class calls the `CopyEnable` method, and the `OnCut` method calls the `OnDelete` method followed by the `OnCopy` method.

The **Delete** menu item is enabled unless the input position is at the end of the document, in which case there is nothing to delete. The `Delete` method is a general method for deleting text and is called when the user presses the *Delete* or *Backspace* keys or when a marked text block is being overwritten.

[PRE36]

The `OnPageBreak` method sets the page break status of the edit paragraph. In case of a page break, the paragraph will be placed at the beginning of the next page. The `OnFont` method displays the standard font dialog that sets the font and color of the next character to be input or the font of the marked block.

[PRE37]

A paragraph can be left, center, right, or justified aligned. The radio mark is present if the paragraph currently edited or all paragraphs currently marked have the alignment in question. All the listeners call the `IsAlignment` and `SetAlignment` methods, which returns the current alignment and sets the alignment, respectively, for the edited paragraph or all marked paragraphs.

[PRE38]

The `OnChar` method is called every time the user presses a graphical character; it calls the `InsertChar` or `OverwriteChar` method, depending on whether the keyboard holds `insert` or `overwrite` mode. When the text is marked and the user changes the font, the font is set on all marked characters. However, when editing text, the font of the next character to be input is set.

When the user does anything else than input the next character, such as clicking the mouse or pressing any of the arrow keys, the `ClearNextFont` method is called, which clears the next font by setting it to the `SystemFont` method.

[PRE39]

The `OnKeyDown` method is called every time the user presses a key, such as the arrow keys, *Page Up* and *Page Down*, *Home* and *End*, *Delete*, or *Backspace*:

[PRE40]

When the user presses the key without pressing the *Shift* key at the same time, the caret is moved. However, when they press the *Shift* key, the marking of the text is changed.

[PRE41]

When the user presses the *Home* or *End* key together with the *Ctrl* key, the caret is placed at the beginning or end of the document. If they also press the *Shift* key, the text is marked.

The reason we use listener instead of regular methods is that all actions involving the *Ctrl* key are interpreted as accelerators by Small Windows. The listeners are also added to a menu in the following constructor.

[PRE42]

There are also the *Return*, *Backspace*, and *Delete* keys, in which case we do not care whether the *Shift* or *Ctrl* key is pressed. The *Delete* key is handled by the **Delete** menu item accelerator.

[PRE43]

When the user moves the caret with the keyboard, the edit character will be visible. The `MakeVisible` method makes sure it is visible, even if it means scrolling the document.

[PRE44]

When something happens to the paragraph (characters are added or deleted, the font or alignment is changed, or the page setup), the positions of the characters need to be calculated. The `GenerateParagraph` method calculates the surrounding rectangle for each of its character and generates its line list by calling the `GenerateSizeAndAscentList` method to calculate the size and ascent line for the characters, the `GenerateLineList` method to divide the paragraph into lines, the `GenerateRegularLineRectList` method to generate the character rectangles for left, center, or right aligned paragraphs or the `GenerateJustifiedLineRectList` method for justified paragraphs, and the `GenerateRepaintSet` method to invalidate the changed characters.

[PRE45]

One central part of this application is the `wordMode` method. At a certain time, the application can be set to `edit` mode (the caret is visible), in which case `wordMode` is the `WordEdit` method, or `mark` mode (a part of the text is marked), in which case `wordMode` is the `WordMark` method. Later in the chapter, we will encounter expressions such as **in edit mode** and **in mark mode**, which refer to the value of `wordMode`: `WordEdit` or `WordMark`.

We will also encounter the expressions **in insert mode** and **in overwrite mode**, which refer to the `input` mode of the keyboard, the `InsertKeyboard` or `OverwriteKeyboard` method, which is returned by the `GetKeyboardMode` method in the Small Windows class `Document`.

The `totalPages` field holds the number of pages, which is used when printing and when setting the vertical scroll bar. The list of characters is stored in the `charList` list, and the list of paragraph pointers is stored in the `paragraphList` list. Note that the paragraphs are dynamically created and deleted `Paragraph` objects while the characters are static `CharInfo` objects. Also note that each paragraph does not hold a character list. There is only one `charList`, which is common to all paragraphs. However, each paragraph holds its own list of `Line` pointers that are local to the paragraph.

In this chapter, we will also encounter expressions such as **the edit character**, which refers to the character with index `editIndex` in the `charList` list. As mentioned at the beginning of this chapter, each character has pointers to its paragraph and line. The expressions **the edit paragraph** and **the edit line** refer to the paragraph and line pointed at by the edit character.

The `firstMarkIndex` and `lastMarkIndex` fields hold the indexes of the first and last marked characters in `mark` mode. They are also referred to in expressions such as **the first marked character**, **the first marked paragraph**, and **the first marked line** as well as **the last marked character**, **the last marked paragraph**, and **the last marked line**. Note that the two fields refer to the chronological order, not necessarily their physical order. When needed, we will define the `minIndex` and `maxIndex` methods to refer to the first and last markings in the document in physical order.

When the user sets the font in `edit` mode, it is stored in the `nextFont` font, which is then used when the user inputs the next character. The caret takes into consideration the status of the `nextFont` font, that is, if the  `nextFont` font is not equal to the `ZeroFont` font, it is used to set the caret. However, the `nextFont` font is cleared as soon as the user does anything else.

The user can zoom the document by menu items or by touching the screen. In that case, we need the `initZoom` and `initDistance` fields to keep track of the zooming. Finally, we need the `WordFormat` field to identify cut, copied, and pasted application-specific information. It is given the arbitrary value of 1002.

[PRE46]

**WordDocument.cpp**

[PRE47]

The `WordDocument` constructor calls the `StandardDocument` constructor. The `UnicodeFormat` and `AsciiFormat` methods are general formats defined by Small Windows, while the `WordFormat` method is specific to this application.

[PRE48]

The **Format** menu holds the **Font** and **Page Break** menu items. Unlike the earlier applications in this book, we send `true` to `StandardFileMenu`. It indicates that we want to include the **Page Setup**, **Print Preview**, and **Print** menu items in the **File** menu.

[PRE49]

The **Alignment** menu holds items for the left, center, right, and justified alignment:

[PRE50]

The `extraMenu` menu is only added for the accelerators; note that we do not add it to the menu bar. The text of the menu, or its items, does not matter either. We only want to allow the user to jump to the beginning or end of the document by pressing the *Ctrl* key with *Home* or *End*, and possibly *Shift*.

[PRE51]

Finally, we call the `InitDocument` method that initializes the empty document. The `InitDocument` method is also called by the `ClearDocument` and `Delete` classes as follows, when the initialization code is placed in its own method.

[PRE52]

A document always holds at least one paragraph, which, in turn, holds at least a newline. We create the first character and the first left-justified paragraph. The paragraph and character are added to the `paragraphList` and `charList` lists.

Then, the paragraph is calculated by the `GenerateParagraph` method and distributed on the document by the `CalculateDocument` method. Finally, the caret is updated by the `UpdateCaret` method.

[PRE53]

## The caret

Since in this chapter we introduce text handling, we need to keep track of the caret: the blinking vertical bar (in `insert` mode) or block (in `overwrite` mode) indicating where to input the character. The `UpdateCaret` method is called by the `OnKeyboardMode` method (which is called when the user presses the *Insert* key) as well as other methods when the input position is being modified.

[PRE54]

In `edit` mode, the caret will be visible, and we obtain the area from the edit character. However, if the `nextFont` font is active (does not equal the `SystemFont` font), the user has changed the font, which we must take into consideration. In that case, we set the width and height of the caret in accordance with the size of an average character of the `nextFont` font.

[PRE55]

If the `nextFont` font is not active, we check whether the keyboard holds `insert` mode and the caret is not located at the beginning of the paragraph. In that case, the caret's vertical coordinates will reflect the font size of the preceding character, since the next character to be input will be given its font.

[PRE56]

If the keyboard holds the `insert` mode, the caret will be a vertical bar, regardless of whether the `nextFont` font is active. It is given the width of one unit (which is later rounded to the width of one physical pixel).

[PRE57]

The caret will not extend outside the page. If it does, its right border is set to the page's border.

[PRE58]

Finally, we need the top position of the edit paragraph, since the caret so far is calculated relative to its top position.

[PRE59]

In `mark` mode, the caret will be invisible. Therefore, we call `ClearCaret` as follows:

[PRE60]

## Mouse input

The `OnMouseDown`, `OnMouseMove`, `OnMouseUp`, and `OnDoubleClick` methods take the pressed buttons and the mouse coordinates. In all four cases, we check that the left mouse button is pressed. The `OnMouseDown` method first calls the `EnsureEditStatus` method in order to clear any potential marked area. Then it sets the application to `mark` mode (which may later be changed by the `OnMouseUp` method) and looks up the index of the character pointed at by calling the `MousePointToIndex` method. The `nextFont` field is cleared by a call to the `ClearNextFont` method. We also call the `UpdateCaret` method, since the caret will be cleared while the user drags the mouse.

[PRE61]

In the `OnMouseMove` method, we retrieve the paragraph and character of the mouse by calling the `MousePointToIndex` method. If the mouse has been moved to a new character since the last call to the `OnMouseDown` or `OnMouseMove` method, we update the marked text by calling the `InvalidateBlock` method with the current and new mouse position, which invalidates the part of the text between the current and previous mouse event. Note that we do not invalidate the whole marked block. We only invalidate the block between the previous and current mouse positions in order to avoid dazzles.

[PRE62]

In the `OnMouseUp` method, we just have to check the last position. If it is the same as the first position (the user pressed and released the mouse at the same character), we change the application to `edit` mode and call the `UpdateCaret` method to make the caret visible.

[PRE63]

The `MousePointToIndex` method finds the paragraph that the user has clicked on and calls the `MousePointToParagraphIndex` method to find the character in the paragraph. The reason we divide the functionality into two methods is that the `MousePointToIndexDown` method in [Chapter 7](ch07.html "Chapter 7. Keyboard Input and Character Calculation"), *Keyboard Input and Character Calculation*, also calls the `MousePointToParagraphIndex` method, which iterates through the paragraph list. If the vertical position is less than the top position of a paragraph, the correct paragraph is the previous one.

This somewhat cumbersome way of finding the correct paragraph is due to the fact that paragraphs are distributed over the pages in such manner that when a paragraph does not fit on the rest of the page, or if it is marked with a page break, it is placed at the beginning of the next page. This may result in parts of the document where no paragraph is located. If the user clicks on such an area, we want the paragraph located before that area to be the correct one. In the same way, if the user clicks below the last paragraph of the document, it becomes the correct one.

[PRE64]

The `MousePointToParagraphIndex` method finds the clicked character in the paragraph. First, we subtract the paragraph's top position from the mouse position, since the paragraph's line coordinates are relative to the paragraph's top position.

[PRE65]

As mentioned previously, the user may click on a position below the paragraph's area. In that case, we set the mouse position to its height, `-1`, which is equivalent to the user clicking on the last line of the paragraph.

[PRE66]

First, we need to find the correct line in the paragraph. We check every line and test if the mouse position is located within the line by comparing it to the sum of the line's top position and its height. Compared to the paragraph search in the `MousePointToIndex` method, as mentioned previously, this search is a bit simpler, since there is no space between the lines in the paragraph as there may be between the paragraphs in the document.

[PRE67]

When we have found the correct line, we have three cases to consider: the user may have clicked on the left of the text (if the paragraph is center or right aligned), to its right (if it is left or center aligned), or on the text itself. If they have clicked on the left or right of the line, we return the index of the first or last character of the line. Note that we add the index of the first character of the paragraph, since the indexes of the lines are relative to the paragraph's first index.

[PRE68]

If the user has clicked on the text, we need to find the correct character. We iterate through the characters of the line and compare the mouse position to the right-hand border of the character. When we have found the correct character, we need to decide whether the user has clicked near the character's left or right border. In case of the right border, we add one to the character index.

[PRE69]

As mentioned previously, there is no space between the lines in a paragraph. Therefore, we will always find the correct line and never reach this point. However, in order to avoid compiler errors, we still have to return a value. In this book, we will on a few occasions use the following notation:

[PRE70]

When the user double-clicks the left mouse button, the word hit by the mouse will be marked. The application has been set to `edit` mode and the `editIndex` method has been properly set, because the call to the `OnDoubleClick` method is always preceded by calls to the `OnMouseDown` and `OnMouseUp` methods. If the mouse hits a word, we mark the word and set the application to `mark` mode.

We find the indexes of the first and last characters in a word by calling the `GetFirstWordIndex` and `GetLastWordIndex` methods. If the first index is less than the last index, the user has double-clicked on an actual word, which we mark. If the first index is not less than the last index, the user has double-clicked on a space or a delimiter, in which case the double-click has no effect.

[PRE71]

In the `GetFirstWordIndex` method, we find the index of the first character of the word by going backward in the character list until we reach the beginning of the document or a character that is not a letter.

[PRE72]

In the `GetLastWordIndex` method, we do not need to check the end of the character list, since the last character always is a newline, which is not a letter. Note that in this case we return the index of the character after the last character of the word, since the marking of text is valid up to, but not inclusive of, the last character.

[PRE73]

## Touchscreen

On a touchscreen, the user can zoom the document by dragging two fingers on the screen. The `OnTouchDown` method is called when the user touches the screen, and the `OnTouchMove` method is called when they move their fingers. Unlike the mouse input methods mentioned previously, the user can touch several points on the screen at the same time. The points are stored in the `pointList` list.

If the list does not hold two points, we just let the `Window` class perform the default action, which is to convert each touch action to a mouse action.

[PRE74]

When the user moves their fingers on the screen, the distance between the fingers is calculated and the zoom is set with regard to the initial distance. The zooming is allowed to range between 10% (factor 0.1) and 1,000% (factor 10.0):

[PRE75]

## Page setup and calculation

The `OnPageSetup` method is called when the user selects the standard **Page Setup** menu item in the **File** menu. Since the page settings have been altered, we need to recalculate each paragraph as well as the whole document.

[PRE76]

A small change may affect the whole document, and we need to calculate the paragraphs and distribute them on the pages in the document.

[PRE77]

We iterate through the paragraph list, and in case the current document height differs from the paragraph's top position, we update its top position and invalidate it.

[PRE78]

We have a page break if the paragraph is marked with a page break and if it is not already located at the top of a page.

[PRE79]

The paragraph does not fit on the rest of the page if its top position plus its height is greater than the page height.

[PRE80]

If we have a page break, or if the paragraph does not fit on the rest of the page, we invalidate the rest of the page and place the paragraph at the top of the next page.

[PRE81]

Since the paragraph has been moved to a new position, we need to invalidate its new area.

[PRE82]

If the paragraph fits on the rest of the document, we just increase the document height.

[PRE83]

After the last paragraph, we need to invalidate the rest of the last page.

[PRE84]

If the number of pages has changed, we invalidate the pages that differ.

[PRE85]

## Painting and drawing

The `OnPaint` method performs the action that is specific to drawing the client area, while the `OnPrint` method performs the action specific to printing. The default behavior for both the `OnPaint` and `OnPrint` methods in the `StandardDocument` class is to call the `OnDraw` method.

In the application of the previous chapters, we have overridden only the `OnDraw` method, resulting in the same drawing regardless of whether the drawing occurs in the client area or is sent to a printer. However, in this application, we also override the `OnPaint` method, which fills the parts of the client area outside the document with a light gray color and places the text **Page Break** between every pair of pages, and finally calls the `OnDraw` method that performs the actual drawing of the document.

[PRE86]

The `OnDraw` method draws every character in the `charList` list. The `drawMode` parameter is `Paint` if the `OnDraw` method is called by the `OnPaint` method, and `Print` if it is called by the `OnPrint` method. In the previous applications, we have ignored the `drawMode` method. However, in this application, we draw a small square at every paragraph marked with a page break, if called by the `OnPaint` method.

[PRE87]

If the character is marked, its text and background colors are inverted.

[PRE88]

If the character is newline, a space is drawn instead.

[PRE89]

If the character's rectangle is located outside the page, its right border is set to the page right border.

[PRE90]

Finally, the character is drawn:

[PRE91]

Actually, there is one more thing: if the `OnDraw` method has been called by the `OnPaint` method, we draw a small red square (2 × 2 millimeters) at its top-left position for every paragraph marked with a page break.

[PRE92]

## File management

The `ClearDocument` method is called by the `StandardDocument` class when the user selects the **New** menu item in the **File** menu; the `WriteDocumentToStream` method is called when they select the **Save** or **Save As** menu items in the **File** menu, and the `ReadDocumentFromStream` method is called when they select the **Open** menu item.

The `ClearDocument` method deletes every paragraph in the `paragraphList` list by calling the `DeleteParagraph` method, which, in turn, deletes each line of the paragraph. This is actually the only memory we need to delete, since it is the only dynamically allocated memory of this application. Finally, the `InitDocument` method is called, which initializes an empty document.

[PRE93]

The `WriteDocumentToStream` method writes all the information about the document to the stream: the `application` mode (edit or mark), the index of the edit character, the indexes of the first and last marked characters, the number of pages in the document, and the next font. The idea is that the document will be opened in the exact same shape as it was written.

[PRE94]

However, if the file suffix is `.txt`, we save the word in text format and discard all formatting.

[PRE95]

The `ReadDocumentFromStream` method reads the information written by the `WriteDocumentToStream` method. Note that the `MakeVisible` method is called at the end in order to make the current position visible.

[PRE96]

However, if the file has the file suffix `.txt`, we just read the characters, and all characters are given the system font.

[PRE97]

## Cut, copy, and paste

The **Copy** item in the **Edit** menu is enabled in `mark` mode:

[PRE98]

As long as the `CopyEnable` method mentioned previously returns `true`, we are always ready to copy in every format. Therefore, we must let the `IsCopyAsciiReady`, `IsCopyUnicodeReady`, and `IsCopyGenericReady` methods return `true` (if they return `false` in the `StandardDocument` class).

[PRE99]

The `CopyAscii` method simply calls the `CopyUnicode` method, since the text is stored in the generic text format and is transformed into ASCII and Unicode when saved to the global clipboard. The `CopyUnicode` method iterates through the marked paragraphs and, for each marked paragraph, extracts the marked text that is stored in the paragraph to the `textList` parameter. When it encounters a newline, it pushes the current text in the `textList` parameter.

[PRE100]

The `CopyGeneric` method is simpler than the `CopyUnicode` method. It first saves the number of characters to be copied, then iterates through the marked characters (not the paragraphs), and then calls the `WriteCharInfoToClipboard` method for each character. This works, since each pair of paragraphs is already separated by a newline in the `charList` list. We really do not care about the format, since there is just one format (`WordFormat`) for generic cut, copy, and paste operations in this application.

[PRE101]

One difference between copying and pasting is that when the user selects **Cut** or **Copy**, the marked text is copied in all three formats (ASCII, Unicode, and generic) given in the preceding `StandardDocument` constructor. Their order does not really matter. When pasting, on the other hand, the `StandardDocument` constructor tries to paste the text in the formats order given in the constructor call. If it finds pasted information in one format in the global clipboard, it does not continue to check the other format. In this application, it means that if there is text copied in the generic format (`WordFormat`), then that text is pasted regardless of whether there is text in the ASCII of Unicode format (`AsciiFormat` or `UnicodeFormat`).

The `PasteAscii` method calls the `PasteUnicode` method (again, both ASCII and Unicode text are transformed into the generic text type), which iterates through the `textList` parameter and inserts a new paragraph for each text. Note that we do not override the `PasteEnable` method, since the `StandardDocument` constructor handles it by checking if there is a clipboard buffer with any of the formats defined in the `StandardDocument` constructor call.

The idea is that the first and last text in text list will be merged by the first and last part of the edit paragraph. The potential remaining text will be inserted as paragraphs in between. First we delete the marked text, if present, ensure `edit` mode, and clear the `nextFont` parameter (setting it to `SystemFont`).

[PRE102]

We remove the edit paragraph from the paragraph list, which makes it easier to insert the pasted paragraphs later on.

[PRE103]

We use the font of the edit character and the alignment of the edit paragraph for the pasted characters and paragraphs.

[PRE104]

We save the number of the remaining characters of the edit paragraph. We also save the current edit index in order to calculate the total number of pasted characters at the end.

[PRE105]

We insert the characters of each text in the edit paragraph.

[PRE106]

Since each text will finish a paragraph, except the last one, we create and insert a new paragraph.

[PRE107]

For the last text, we use the original edit paragraph and change its last character index.

[PRE108]

We may also need to update the index of the succeeding paragraphs, since more than one paragraph may have been pasted. Since we know that at least one character has been pasted, we certainly need to at least modify the first and last index of the succeeding paragraphs.

[PRE109]

The `PasteGeneric` method reads and inserts the generic paragraph information stored in the clipboard in a way similar to the preceding `PasteUnicode` method. The difference is that the paragraphs are separated to be newlines and that each pasted character comes with its own font.

[PRE110]

We erase the edit paragraph in order to make the insertion easier, just as in the preceding `PasteUnicode` method. We use the alignment of the edit paragraph, but not the font of the edit character since each pasted character has its own font.

[PRE111]

We read the paste size, which is the number of character to be pasted.

[PRE112]

We read each character from the paste buffer and insert the characters into the character list. When we encounter a newline, we insert a new paragraph.

[PRE113]

We need to calculate the original paragraph before we insert it.

[PRE114]

Similar to the preceding `PasteUnicode` case, we may need to update the index of the succeeding paragraphs, since more than one paragraph may have been pasted. We also need to modify their first and last index, since at least one character has been pasted.

[PRE115]

## Delete

In `edit` mode, it is possible to delete a character unless it is located at the very end of the document. In `mark` mode, the marked text can always always be deleted:

[PRE116]

In `edit` mode, we delete the edit character, and in `mark` mode, we delete the marked text. In both cases, we call the `Delete` method to perform the actual deleting.

[PRE117]

The `Delete` method is called by the `OnDelete`, `EnsureEditStatus`, `PasteUnicode`, and `PasteGeneric` methods. It removes the characters between the given indexes, which do not have to be in order. The removed paragraphs are deleted and the succeeding paragraphs are updated.

[PRE118]

The deleted area covers at least two paragraphs, we set the characters of the maximal paragraph to point at the minimal paragraph, since they will be merged. We also set their rectangles to zero, to ensure that they will be redrawn.

[PRE119]

The characters are removed from the `charList` list and the last index of the minimal paragraph is updated. It is set to the last character of the maximal paragraph (that may be the same paragraph as the minimal paragraph) minus the number of the characters to be deleted. The minimal paragraph is then regenerated.

[PRE120]

The paragraphs between the minimal and maximal paragraphs, if any, are deleted and the indexes of the succeeding paragraphs are set. We call `DeleteParagraph` for each paragraph to delete their dynamically allocated memory.

[PRE121]

Finally, we need to set the indexes of the succeeding paragraphs. Note that we have to update the first and last index regardless of whether any paragraphs have been removed, since we have removed at least one character.

[PRE122]

When the delete process is finished, the application is set to `edit` mode, and the edit index is set to the first marked character.

[PRE123]

## Page break

The **PageBreak** menu item is enabled in `edit` mode, and the `OnPageBreak` method is also quite simple. It just inverses the page break status of the edit paragraph:

[PRE124]

## Font

The `OnFont` method is called when the user selects the **Font** menu item and it displays the font dialog. In `edit` mode, we first need to find the default font to use in the dialog. If the `nextFont` parameter is active (does not equal `SystemFont`), we use it. If it is not active, we check whether the edit character is the first character in the paragraph. If it is the first character, we use its font. If it is not the first character, we use the font of its preceding character. This is the same procedure as in the preceding `UpdateCaret` method:

[PRE125]

If the user closes the font dialog by choosing **Ok**, we set the `nextFont` parameter and recalculate the edit paragraph.

[PRE126]

In `mark` mode, we choose the font of the marked character with the lowest index to be the default font in the font dialog.

[PRE127]

If the user chooses **Ok**, we set the font of every marked character and recalculate each of their paragraphs.

[PRE128]

## Alignment

All the radio alignment listeners call the `IsAlignment` method, and all selection listeners call the `SetAlignment` method.

[PRE129]

In `edit` mode, the `IsAlignment` method checks whether the edit paragraph has the given alignment. In `mark` mode, it checks if all partly or completely marked paragraph have the given alignment. This implies that if several paragraphs are marked with different alignments, no alignment menu item will be marked with a radio button.

[PRE130]

The `SetAlignment` method sets the alignment of the edited or marked paragraphs. In `edit` mode, we just set the alignment of the edit paragraph. Remember that this method can only be called when the paragraph has another alignment. In `mark` mode, we traverse the marked paragraphs and set the alignment on those paragraphs that do not have the alignment already in question. Also remember that this method can only be called if at least one paragraph does not hold the alignment in question. The paragraphs that have changed alignment need to be recalculated. However, the new alignment does not affect the height of the paragraph, which implies that we do not need to call the `CalculateDocument` method for the remaining paragraphs.

[PRE131]

# Summary

In this chapter, you started to develop a word processor capable of handling individual characters. The word processor supports the following:

*   Individual font and style of each character
*   Left, center, right, and justified alignment of each paragraph
*   Paragraphs that are distributed over the pages
*   Scrolling and zooming
*   Touchscreen
*   Cut, copy, and paste with ASCII or Unicode text, as well as application-specific generic information

In [Chapter 7](ch07.html "Chapter 7. Keyboard Input and Character Calculation"), *Keyboard Input and Character Calculation*, we will continue with the keyboard input and character calculation.