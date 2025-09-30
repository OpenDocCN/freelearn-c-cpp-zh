# Chapter 9. Formula Interpretation

The spreadsheet program is capable of handling text, numerical values, and formulas composed by the four arithmetic operators. In order to do so, we need to interpret the formulas. We also need to find the sources of a formula (the cells referred to in the formula) and the targets of a cell (the cells affected by a change).

In this chapter, we will take a look at the following topics:

*   Interpretation (scanning and parsing) of numerical expressions
*   Parse and syntax trees
*   Evaluation of formulas
*   References and matrices
*   Drawing of cells
*   Loading and saving of cells

In the following spreadsheet, the `C3` cell is being edited:

![Formula Interpretation](img/B05475_09_01.jpg)

# Formula interpretation

The core of a spreadsheet program is its ability to interpret formulas. When the user inputs a formula in a cell, it is interpreted and its value is evaluated. The process of formula interpretation is divided into three separate steps. First, given the input string, the **Scanner** generates a **Token List**, then the **Parser** generates a **Syntax Tree**, and the **Evaluator** determines the value.

![Formula interpretation](img/B05475_09_02.jpg)

A token is the least significant part of the formula. For instance, *a1* is interpreted as a reference and *1.2* is interpreted as a value. Assuming that the cells have values according to the following sheet, the formula interpretation process will be as follows. Remember that a formula is text beginning with an equal sign (**=**).

![Formula interpretation](img/B05475_09_03.jpg)![Formula interpretation](img/B05475_09_04.jpg)![Formula interpretation](img/B05475_09_05.jpg)

## The tokens

The scanner takes a string as input and finds its least significant parts-its tokens. Spaces between the tokens are ignored, and the scanner makes out no difference between capital and small letters. The `Value` token needs an extra piece of information to keep track of the actual value, which is called an **attribute**. In the same way, `Reference` needs an attribute to keep track of reference. In this application, there are nine different tokens:

**Token.h**

[PRE0]

| **Token** | **Description** |
| `Plus`, `Minus`, `Star`, and `Slash` | These are the four arithmetic operators: "`+`", "`-`", "`*`", and "`/`" |
| `LeftParenthesis` and `RightParenthesis` | These are the left and right parentheses: "`(`" and "`)`" |
| Value | This is a numerical value, for instance, `124`, `3.14`, or `-0.23`. It does not matter whether the value is integral or decimal. Nor does it matter if the decimal point (if present) is preceded or succeeded by digits. However, the value must contain at least one digit. This needs a value of type double as an attribute. |
| Reference | This is a reference, for instance, `b8, c6`. This needs `Reference` object as an attribute. |
| EndOfLine | This is at the end of the line, there are no more (non-space) characters in the string. |

As stated previously, the string *1.2 * (b2 + c3)* generates the tokens in the table on the next page. The end-of-line token is added at the end of the list.

| **Text** | **Token** | **Attribute** |
| 1.2 | Value | 1.2 |
| * | Star |  |
| ( | LeftParenthesis |  |
| b2 | Reference | row `1`, col `1` |
| + | Plus |  |
| c3 | Reference | row `2`, col `2` |
| ) | RightParanthesis |  |
|  | EndOfLine |  |

The tokens are defined in the `Token` class. A token is made up of a token identifier, a double value in case of the value token, and a `Reference` object in case of the reference token.

**Token.h**

[PRE1]

**Token.cpp**

[PRE2]

## The tree node

As mentioned earlier, the parser generates a syntax tree. More specifically, it generates an object of the `Tree` class (described in [Chapter 12](ch12.html "Chapter 12. The Auxiliary Classes"), *Auxiliary Classes*), which is a template class with a node type: `TreeNode`. There are 10 identities for a node and, similar to `Token`, a value node has a double value as its attribute and a reference node has a reference object as attribute.

**TreeNode.h**

[PRE3]

The default constructor is used when reading the value from a file or the clipboard buffer.

[PRE4]

A cell of a spreadsheet can be saved to a file as well as cut, copied, and pasted, thus we included the following methods:

[PRE5]

The identity and value of the node can only be inspected, not modified. However, the reference can be modified, since it is updated when the user copies a cell and then pastes it to another location:

[PRE6]

**TreeNode.cpp**

[PRE7]

The node identity, the value, and the reference are written and read, as follows:

[PRE8]

## The Scanner – Generating the list of tokens

The task of the `Scanner` class is to group characters into tokens. For instance, *12.34* is interpreted as the value *12.34*. The constructor takes a string as parameter while `Scan` generates a list of tokens by repeatedly calling `NextToken` until the string is empty.

**Scanner.h**

[PRE9]

The `NextToken` method returns `EndOfLine` when it encounters the end of the string. The `ScanValue` and `ScanReference` methods return `true` if they encounter a value or a reference:

[PRE10]

The next token is continually read from the buffer until it is empty:

[PRE11]

**Scanner.cpp**

[PRE12]

`TEXT('\0')` is added to the string for simplicity; instead of checking whether the remaining text is empty, we look for the `null` character:

[PRE13]

The `Scan` method adds the token from the buffer to `tokenList` until it encounters `EndOfLine`. Finally, the list is returned:

[PRE14]

The `NextToken` method does the actual work of the scanner by finding the next token in the buffer. First, we skip the blanks. It is rather simple to extract the token when it comes to the arithmetic symbols and the parentheses. We just check the next character of the buffer. It becomes slightly more difficult when it comes to numerical values or references. We have two auxiliary methods for that purpose: `ScanValue` and `ScanReference`. Take a look at the following code:

[PRE15]

If none of the trivial cases apply, the token may be a value or a reference. The `ScanValue` and `ScanReference` methods find out if that is the case. If not, the scanner has encountered an unknown character and a syntax error exception is thrown:

[PRE16]

`ScanValue` uses the `_stscanf_s` standard function, which is the safe generic version of `sscanf`. The returned value is stored in `fieldCount`, which is set to `1` if the double value was successfully read. We also need the number of the character read, which is stored in `charCount`, in order to erase the correct number of characters from the buffer:

[PRE17]

`ScanReference` checks whether the first two characters are a letter and a digit. If so, it extracts the column and the row of the reference:

[PRE18]

We extract the column by subtracting the lowercase letter from *a*, which gives that the first column has the index zero, and erases the letter from the buffer.

[PRE19]

Similar to `ScanValue`, we extract the row by calling `_stscanf_s`, which reads the row integer value and the number of characters, which we use to erase the characters read from the buffer:

[PRE20]

## The parser – Generating the syntax tree

The user inputs a formula beginning with an equal sign (**=**). The parser's task is to translate the scanner's token list into a syntax tree. The syntax of a valid formula can be defined by a **grammar**. Let's start with a grammar that handles expressions that make use of the arithmetic operators:

![The parser – Generating the syntax tree](img/B05475_09_06.jpg)

A grammar is a set of rules. In the preceding grammar, there are eight rules. **Formula** and **Expression** are called **non-terminals**; **EndOfLine**, **Value**, and the characters **+**, **-**, *****, **/**, **(**, and **)** are called **terminals**. Terminals and non-terminals are called symbols. One of the rules is the grammar's **start rule**, in our case the first rule. The symbol to the left of the start rules is called the grammar's **start symbol**, in our case **Formula**.

The arrow can be read as "**is**", and the preceding grammar can be read as:

*A formula is an expression followed by end-of-line. An expression is the sum of two expressions, the difference of two expressions, the product of two expressions, the quotient of two expressions, an expression enclosed by parentheses, a reference, or a numerical value.*

This is a good start, but there are a few problems. Let's test if the string *1 + 2 * 3* is accepted by the grammar. We can test that by doing a **derivation**, where we start with the start symbol `Formula` and apply the rules until there are only terminals. The digits in the following derivation refer to the grammar rules:

![The parser – Generating the syntax tree](img/B05475_09_07.jpg)

The derivation can be illustrated by the development of a **parse tree**.

![The parser – Generating the syntax tree](img/B05475_09_08.jpg)Let's try another derivation of the same string, with the rules applied in a different order.![The parser – Generating the syntax tree](img/B05475_09_09.jpg)

This derivation generates a different parse tree, which is as follows:

![The parser – Generating the syntax tree](img/B05475_09_10.jpg)

The grammar is said to be ambiguous as it can generate two different parse trees for the same input string, which we would like to avoid. The second tree is obviously a violation of the laws of mathematics, stating that multiplication has higher precedence than addition, but the grammar does not know that. One way to avoid ambiguity is to introduce a new set of rules for each level of precedence:

![The parser – Generating the syntax tree](img/B05475_09_11.jpg)

The new grammar is not ambiguous. If we try our string with this grammar, we can only generate one parse tree, regardless of the order that we choose to apply the rules. There are formal methods to prove that the grammar is not ambiguous; however, that is outside the scope of this book. Check out the references at the end of this chapter for references.

![The parser – Generating the syntax tree](img/B05475_09_12.jpg)

This derivation gives the following tree. As it is not possible to derive two different trees from the same input string, the grammar is **unambiguous**.

![The parser – Generating the syntax tree](img/B05475_09_13.jpg)

We are now ready to write a parser. Essentially, there are two types of parsers: **top-down parser** and **bottom-up parser**. As the terms imply, a top-down parser starts by the grammar's start symbol together with the input string, and it tries to apply rules until we are left with only terminals. A bottom-up parser starts with the input string and tries to apply rules backward, reducing the rules until we reach the start symbol.

It is a complicated matter to construct a bottom-up parser. It is usually not done manually; instead, there are **parser generators** constructing a **parser table** for the given grammar and skeleton code for the implementation of the parser. However, the theory of bottom-up parsing is outside the scope of this book.

It is easier to construct a top-down parser than a bottom-up parser. One way to construct a simple, but inefficient, top-down parser would be to apply all possible rules in random order. If we reach a dead end, we simply backtrack and try another rule. A more efficient, but rather simple, parser is a look-ahead parser. Given a suitable grammar, we only need to look at the next token in order to uniquely determine the rule to apply. If we reach a dead end, we do not have to backtrack; we simply draw the conclusion that the input string is incorrect according to the grammar-it is said to be **syntactically incorrect**, that is, it has a **syntax error**.

The first attempt to implement a look-ahead parser could be to write a function for each rule in the grammar. Unfortunately, we cannot do that quite yet because that would result in a function `Expression` like this:

[PRE21]

Do you see the problem? The method calls itself without changing the input stream, which would result in an infinite number of recursive calls. This is called **left recursion**. We can solve the problem, however, with the help of a simple translation.

![The parser – Generating the syntax tree](img/B05475_09_14.jpg)

The preceding rules can be translated to the equivalent set of rules (where epsilon ε denotes empty string):

![The parser – Generating the syntax tree](img/B05475_09_15.jpg)

If we apply this transformation to the **Expression** and **Term** rules in the preceding grammar, we receive the following grammar:

![The parser – Generating the syntax tree](img/B05475_09_16.jpg)

Let's try this new grammar with our string *1 + 2 * 3*.

![The parser – Generating the syntax tree](img/B05475_09_17.jpg)

The derivation generates the following parse tree:

![The parser – Generating the syntax tree](img/B05475_09_18.jpg)

The requirement for a grammar to be suitable for a look-ahead parser is that every set of rules with the same left-hand side symbol must begin with different terminals at its right-hand side. If it does not have an empty rule, it may have at the most one rule with a non-terminal as the first symbol on the right-hand side. The preceding grammar we covered meets these requirements.

Now we are ready to write the parser. However, the parser should also generate some kind of output, representing the string. One such representation is the **syntax tree**, which can be viewed as an abstract parse tree-we keep only the essential information. For instance, the previous parse tree has a matching syntax, which is as follows:

![The parser – Generating the syntax tree](img/B05475_09_19.jpg)

The following is the `Parser` class. The idea is that we write a method for every set of rules with the same left-hand symbol. Each such method generates a part of the resulting syntax tree. The constructor takes the text to parse and lets the scanner generate a list of tokens. Then, `Parse` starts the parsing process, and returns the generated syntax tree. If an error occurs during the parsing process, a syntax error exception is thrown. When the token list has been parsed, we should make sure that there are no extra tokens left in the list except `EndOfLine`. Also, if the input buffer is completely empty (the user inputs only a single equal sign), there is still the `EndOfLine` token in the list.

The result of the parsing is a syntax tree representing the formula. For instance, the formula *a1 * c3 / 3.6 + 2.4 * (b2 - 2.4)* generates the following syntax tree, and we take advantage of the `Tree` class of [Chapter 12](ch12.html "Chapter 12. The Auxiliary Classes"), *Auxiliary Classes*.

![The parser – Generating the syntax tree](img/B05475_09_20.jpg)

As mentioned in the `TreeNode` section earlier, there are nine types of syntax tree: the four arithmetic operators, unary addition and subtraction, expressions in parentheses, references, and numerical values. We do not actually need the parentheses to store the formula correctly, as the priority of the expression is stored in the syntax tree itself. However, we need it to regenerate the original string from the syntax tree when written in a cell.

**Parser.h**

[PRE22]

The `Parse` method is called in order to interpret the text that the user has input. It receives the token list from the scanner, which holds at least the `EndOfLine` token and parses the token list and receives a pointer to the syntax tree. When the token list has been parsed, it checks whether the next token is `EndOfLine` to make sure that there are no extra characters (except spaces) left in the buffer:

**Parser.cpp**

[PRE23]

The `Match` method is used to match the next token in the list with the expected token. If they do not match or if the token list is empty, a syntax error exception is thrown. Otherwise, the next token is removed from the list:

[PRE24]

The rest of the methods implement the grammar we discussed earlier. There is one method for each for the symbols `Expression`, `NextExpression`, `Term`, `NextTerm`, and `Factor`:

[PRE25]

The `NextExpression` method takes care of addition and subtraction. If the next token is `Plus` or `Minus`, we match it and parse its right operand. Then, we create and return a new syntax tree with the operator in question. If the next token is neither `Plus` nor `Minus`, we just assume that another rule applies and return the given left syntax tree:

[PRE26]

The `NextTerm` method works with multiplication and division in a way similar to `NextExpression`. Remember that we need a set of methods for each precedence level of the grammar.

[PRE27]

The `Factor` method parses values, references, and expressions enclosed by parentheses. If the next token is a unary operator (plus or minus), we parse its expression and create a syntax tree holding the expression:

[PRE28]

If the next token is a left parenthesis, we match it, parse the following expression, and match the closing right parenthesis:

[PRE29]

If the next token is a reference, we receive the reference attribute with its row and column and match the reference token. We create a new syntax tree holding a reference. Note that the parser does not check whether the reference is valid (refers to a cell inside the spreadsheet); that is the task of the evaluation of the formula's value:

[PRE30]

If none of the preceding tokens applies, the user has input an invalid expression and a syntax error exception is thrown:

[PRE31]

# Matrix and reference

The `Matrix` class is used when storing the cells of spreadsheet, and the `Reference` class is used when accessing cells in the spreadsheet.

## The reference class

The `Reference` class holds the row and column of a cell in the `Matrix` class, as shown in the next section:

**Reference.h**

[PRE32]

The default constructor initializes the row and column to zero. A reference can be initialized by and assigned to another reference:

[PRE33]

The compare operators first compare the rows. If they are equal, the columns are then compared:

[PRE34]

The addition operators add and subtract the rows and columns separately:

[PRE35]

The `Clear` method sets both the row and column to zero, and `IsEmpty` returns `true` if the row and column is zero:

[PRE36]

The `ToString` method returns a string representing the reference:

[PRE37]

A reference is inside a block of references defined by a smallest and a largest reference if it is greater than or equal to the smallest one and less than or equal to the largest one:

[PRE38]

The reference can be written to and read from a file stream, the clipboard, and the registry:

[PRE39]

The row and column are inspected by the constant methods and modified by the non-constant methods:

[PRE40]

**Reference.cpp**

[PRE41]

The `ToString` method returns to reference as a string. We increase the number of rows by one, implying that row zero corresponds to *1*. The column is converted to characters, implying that column zero corresponds to *a*. If the number of rows or columns is less than zero, `?` is returned:

[PRE42]

When communicating with the registry, we use the `WriteBuffer` and `ReadBuffer` static methods. In order for that to work, we place the row and column values in the `ReferenceStruct` structure:

[PRE43]

## The Matrix class

The `Matrix` class holds a set of cells organized in rows and columns.

**Matrix.h**

[PRE44]

The matrix can be initialized by or assigned to another matrix; in both cases, they call `Init` to do the actual initialization:

[PRE45]

The index operator takes a row or a `Reference` object. In the case of a row, an array of columns is returned (technically, the address of its first value is returned), which can be further indexed by the regular index operator to obtain the value in the buffer. In the case of a reference, the value is accessed directly by indexing the row and column of the buffer. Note that in this class, the vertical row coordinate holds the first index and the horizontal column coordinate the second index:

[PRE46]

Since Matrix is a template class, we place the definition of its methods in the `header` file. The default constructor lets the default cell constructor initialize the cells:

[PRE47]

The copy constructor and the assignment operator copies the cells by calling `Init`:

[PRE48]

# The cell

The cell can hold three modes: (possible empty) text, a numerical value, or a formula. Its mode is stored in the `cellMode` field. It can hold the value `TextMode`, `ValueMode`, or `FormulaMode`. Similar to `CalcDocument` in this chapter and `WordDocument` in the previous chapters, we refer to the current value of `cellMode` in expressions such as **in text mode**, **in value mode**, and **in formula mode**.

`HeaderWidth`, `HeaderHeight`, `ColWidth`, and `RowHeight` are the size of the headers and cells of the spreadsheet. In order for the cell text to not overwrite the cell's borders, `CellMargin` is used. The spreadsheet is made up of ten rows and four columns.

**Cell.h**

[PRE49]

A cell can be aligned at the left, center, right or justified in the horizontal direction, and it can be aligned at the top, center, or bottom in the vertical direction:

[PRE50]

The `Clear` method is called when the user selects the new menu item and clears the font and background color of the cell before calling `Reset`, which clears the text and sets the cell to the text mode. `Reset` is also called when the user deletes the cell, in that case, the text is cleared, but not the font or color:

[PRE51]

The `CharDown` method is called when the user inputs a character that is inserted before the current character or overwrites it depending on the value of the `keyboardMode` parameter. When the user double-clicks on the text in a cell, `MouseToIndex` calculates the index of the character clicked on:

[PRE52]

The `Text` and `CaretList` methods return the text and caret rectangle list of the cell.

[PRE53]

The font and background color of the cell can both be modified and inspected, so can the horizontal and vertical alignment:

[PRE54]

The `DrawCell` method draws the border of the cell in black, fills the cell with the background color, and draws the text. All colors are inverted if the inverse parameter is true, which it is if the cell is either being edited or is marked:

[PRE55]

The `DisplayFormula` method is called when the user starts editing the cell. A cell with a formula can be displayed with its value or its formula. When the user edits the cell, the formula is displayed. When they mark it, its value is displayed. The `DisplayFormula` method replaces the value by the formula (or an error message in case of an incorrect formula):

[PRE56]

The `InterpretCell` method interprets the text of the cell, which is interpreted as text, a numerical value, or a formula. If the formula contains a syntax error, an exception is thrown:

[PRE57]

In the `formula` mode, `GenerateSourceSet` analyzes the formula and returns the (possibly empty) set of all its references. In the `text` or `value` mode, an empty set is returned:

[PRE58]

In the `formula` mode, `TreeToString` returns the formula converted from the syntax tree to the string that is displayed in the cell when being edited:

[PRE59]

When the user cuts, copies, and pastes cells, their references are updated. `UpdateTree` updates all references in the formula mode:

[PRE60]

The `HasValue` method returns `true` if the cell holds a value: `true` in the `value` mode, `false` in the `text` mode, and `true` in the `formula` mode if it has been evaluated to a value, `false` if an evaluation error (missing value, reference out of scope, circular reference, or division by zero) occurred:

[PRE61]

The `Evaluate` method evaluates the syntax tree of the formula; `valueMap` holds the values of the cells in the source set:

[PRE62]

The cell can be saved to a file or cut, copied, and pasted:

[PRE63]

As mentioned at the beginning of this section, the cell can hold (possibly empty) text, a numerical value, or a formula, indicated by the value `cellMode`:

[PRE64]

All characters in the cell hold the same font and background color. The cell can be aligned at the left, center, right, or justified horizontally, and it can be aligned at the top, center, or bottom vertically:

[PRE65]

The `text` field holds the text displayed in the cell. In the `edit` mode, it is the text currently input by the user. In the `mark` mode, it is the text input by the user (in text mode), a numerical value input by the user converted to text, the calculated value of a formula, or an error message (missing value, reference out of scope, circular reference, or division by zero):

[PRE66]

The caret list holds the caret rectangle of each character in `text`. It also holds the rectangle for the index after the last character, which means that the size of the caret list is always one more than the text:

[PRE67]

When the value of a formula is being calculated, it may result in a value or any of the errors we discussed earlier. If the cell holds a value, `hasValue` is `true` and `value` holds the actual value:

[PRE68]

When the user inputs a formula starting with *=*, it is interpreted as a syntax tree by the `Scanner` and `Parser` classes, and it is stored in `syntaxTreePtr`:

[PRE69]

**Cell.cpp**

[PRE70]

The width of a cell is the width of the column minus the margins, and its height is the row height minus the margins:

[PRE71]

When a cell is created, it is empty, it holds the text mode, it is center aligned in both horizontal and vertical directions, and it holds the system font with black text on white background:

[PRE72]

The copy constructor and assignment operator check whether `syntaxTreePtr` is `null`, if it is not null it is copied dynamically, its constructor continues copying its children recursively. It is not enough to simply copy the pointer, since one of the formulas of either the original or copy cell may be changed, but not the other one:

[PRE73]

One difference between the copy constructor and the assignment operator is that we delete the syntax tree pointer in the assignment operator since it may point at dynamically allocated memory, which is not the case in the copy constructor. If it points at `null`, the `delete` operator does nothing:

[PRE74]

The syntax tree is the only dynamically allocated memory of the cell. Again, in case of a null pointer, `delete` does nothing:

[PRE75]

The difference between `Clear` and `Reset` is:

*   `Clear` is called when the user selects the **New** menu item and the spreadsheet shall be totally cleared and also the cell's font, color and alignment shall be reset.
*   `Reset` is called when the user deletes a cell and its mode and text shall be reset.

[PRE76]

## Character input

The `CharDown` method is called by `WindowProc` (which in turn is called by the Windows system) every time the user presses a graphical character. If the input index is at the end of the text (one step to the right of the text), we just add the character at the end. If it is not at the end of the text, we have to take into consideration the keyboard mode, which is either insert or overwrite.

In case of an insert, we insert the character, and in case of overwrite, we overwrite the character previously located at the edit index. Unlike the word processor in the previous chapters, we do not have to deal with the font, since all characters in the cell have the same font:

[PRE77]

The `MouseToIndex` method is called when the user double-clicks on the cell. First, we need to subtract the cell margin from the mouse position, then we iterate the caret list and return the position of the character hit by the mouse. If the user hits to the left of the first character (aligned at the center or right), zero index is returned, and if they hit to the right of the last character (aligned to the left or center), the size of the text is returned, which corresponds to the index to the right of the last character:

[PRE78]

## Drawing

The `Draw` method is called when the contents of the cell are to be drawn. The drawing of the text is rather straightforward-for each character in the character list, we just draw the character in its caret rectangle. This particular cell may be marked or in the process of being edited, in which case the inverse is true. In that case, the text, background, and border colors are inverted. In order to not overwrite the border of the cell, we also take the cell margin into consideration:

[PRE79]

## Caret rectangle list generation

When the user adds or removes a character of the text of a cell or changes its font or alignment, the caret rectangles need to be recalculated. `GenerateCaretList` can be considered a simplified version of `GenerateParagraph` in the word processor of the previous chapters. Its task is to calculate the character rectangles, which are used when setting the caret, drawing the text, and calculating the index of a mouse click.

First, we need to calculate the width of each character as well as the width of the text in order to set its horizontal start position. In case of justified alignment, we calculate the text width without spaces and count the spaces:

[PRE80]

When we have calculated the text width, we set the horizontal start position. In case of left or justified alignment, the start position is set to the cell margin. In the case of justified alignment, we also set the width of each space in the text. In the case of right alignment, we add the difference between the width of the cell and the text to the cell margin in order to place the rightmost part of the text at the right border in the cell. In the case of center alignment, we add half the difference in order for the text to be placed in the middle of the cell:

[PRE81]

The vertical top position is set in a similar manner. In the case of top alignment, the top position is set to the cell margin. In the case of bottom alignment, we add the difference between the height of the cell and the text to the cell margin in order to place the bottom part of the text at the bottom border in the cell. In the case of center alignment, we add half the difference in order to place the text in the middle of the cell:

[PRE82]

When the horizontal start position and the top vertical position has been set, we iterate through the characters and add the rectangles to `caretList` for each of them. Note that we use the value of `spaceWidth` for spaces in the case of justified alignment:

[PRE83]

When each rectangle is added, we add the rectangle for the character to the right of the text. We set its width to the width of an average character of the cell's font:

[PRE84]

## Formula interpretation

When the user single-clicks or double-clicks on a cell, its text remains unchanged in the text or value mode, but it gets changed in the formula mode. In the formula mode, the calculated value of the formula is displayed in the mark mode, while in the edit mode, the formula itself is displayed. `DisplayFormula` calls `TreeToString` in the formula mode, which generates the text of the formula:

[PRE85]

The `InterpretCell` method is called when the user terminates the text input by pressing the ***Enter*** or ***Tab*** key or clicking the mouse. If the user has input a formula (starting with *=*), it is parsed. `Parse` returns a syntax tree holding the formula or throws an exception in the case of a syntax error. Note that `InterpretCell` only report the syntax error. All other errors (missing value, references out of range, circular reference, or division by zero) are handled by the following `Evaluate`:

[PRE86]

The `GenerateSourceSet` method traverses the syntax tree and extracts a (possible empty) set of all its references in the formula mode. In the case of text or value mode, the set is empty, since only formulas hold references:

[PRE87]

In case of unary addition or subtraction or an expression enclosed by parentheses, the source set of its child node is returned:

[PRE88]

In the case of a binary expression, the union of the source sets of the two children is returned:

[PRE89]

In the case of a reference, a set holding only the reference is returned if it is located in the spreadsheet. No references outside the spreadsheet are included in the set:

[PRE90]

Finally, in the case of a value, an empty set is returned:

[PRE91]

The `TreeToString` method traverses the syntax tree and converts it to a string. Note that it is quite possible to have a formula with a reference out of scope. However, the `Reference` class returns `?` in that case:

[PRE92]

In the case of unary addition or subtraction, `+` or `-` is added to the text of the child node:

[PRE93]

In the case of a binary expressions `+`, `-`, `*`, or `/` is inserted between the text of the child nodes:

[PRE94]

In the case of an expression enclosed by parentheses, the text of the child node enclosed by parentheses is returned:

[PRE95]

In the case of a reference, its text is returned. Again, if the reference is out of range, `?` is returned:

[PRE96]

In the case of a value, its converted text is returned:

[PRE97]

When the user copies and pastes a block of cells, the references of each formula are relative and will be updated. `UpdateTree` looks for and updates references in the syntax tree. In all other cases, it iterates through the child list and calls `UpdateTree` recursively for each child (one child each in a unary expression and a parentheses expression, two children in a binary expression, and no children in values or references):

[PRE98]

When the value of a formula is evaluated, it may return a valid value, in which case `hasValue` is set to `true`. However, if an error occurs during the evaluation (missing value, references out of range, circular reference, or division by zero), `hasValue` is set to `false`. `hasValue` is called when a value of a formula of another cell is being evaluated. If it returns `false`, the evaluation will result in the missing value error:

[PRE99]

In the formula mode, the formula is being evaluated to a value. If an error occurs (missing value, reference out of range, circular reference, or division by zero), an exception is thrown by `Evaluate`, and the cell text is set to the error message text. Note that it is possible to input references out of scope, which `InterpretCell` accepts. However, `Evaluate` throws an exception with an error message that is displayed in the cell.

Moreover, it is quite possible to cut, copy, and paste a cell so that its references get located out of the scope and then cut, copied, and pasted again so that the references become valid. However, if the user edits a formula with references out of the scope, `?` is returned by the `ToString` method in the `Reference` class, since it is difficult to express references with negative columns:

[PRE100]

The `Evaluate` method finds the current value of the cell by looking up the values of the cells referred to by the formula:

[PRE101]

In the case of a unary or binary expression, the value is calculated (unary addition is only present for the sake of completeness and does not change the value):

[PRE102]

In case of division by zero, an exception is thrown.

[PRE103]

In the case of an expression within parentheses, we simply return its evaluated value:

[PRE104]

In the case of a reference, we look up the source cell in `valueMap`. In the case of a source cell with a missing value (not present in `valueMap`) or a reference out of scope (referring to a cell outside the spreadsheet), exceptions are thrown:

[PRE105]

In the case of a value, we simply return the value:

[PRE106]

## File management

The `WriteDocumentToStream` method is called by `CalcDocument` every time the user selects the **Save** or **Save As** menu items from the file menu. In the formula mode, we call `WriteTreeToStream` on the syntax tree:

[PRE107]

In `ReadCellFromStream`, we dynamically create and read the syntax tree in the formula mode:

[PRE108]

The `WriteCellToClipboard` and `ReadCellFromClipboard` methods are called by `CalcDocument` when the user cuts, copies, and pastes the cell. It works in the same way as `WriteDocumentToStream` and `ReadCellFromStream` we saw earlier:

[PRE109]

# Further reading

If the scanner and parser of this chapter have got you interested in compilers, I recommend that you refer to *Compilers: Principles, Techniques, and Tools* by A. V. Aho et al. (second edition. Addison Wesley, 2007). It is the second edition of the classic *Dragon Book*. The authors explain the theory and practice of compilers from scanning and parsing to advanced optimization.

If the concept of graphs has caught your interest, I recommend *Introduction to Graph Theory* by D. B. West (Prentice Hall, 2000), which reasons about graphs from a mathematical point of view.

# Summary

In this chapter, we covered the spreadsheet program implementation. This chapter concludes the first part of this book: how to develop an application with Small Windows. [Chapter 10](ch10.html "Chapter 10. The Framework"), *The Framework*, introduces the second part: the implementation of Small Windows.