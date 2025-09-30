# Chapter 3. Working with Labels

In this chapter, we're going to create labels. To display labels on the screen, you can use the `Label` class with system fonts, true type fonts, and bitmap fonts. The following topics will be covered in this chapter:

*   Creating system font labels
*   Creating true type font labels
*   Creating bitmap font labels
*   Creating rich text

# Creating system font labels

Firstly, we will explain how to create a label with system fonts. System fonts are the fonts already installed on your devices. Since they are already installed, there is no need to go through the installation process. Therefore we will skip the installation instructions for system fonts in this recipe, and dive directly into creating labels.

## How to do it...

Here's how to create a label by specifying a system font. You can create a single-line label by using the following code:

[PRE0]

![How to do it...](img/B00561_03_01.jpg)

## How it works...

You should use the `Label` class to display strings by specifying a string, a system font, and the font size. The `Label` class will display a string that is converted into an image. After creating a `Label` instance, you can use it in the same way as you use `Sprite`. Because `Label` is also a Node, we can use properties such as actions, scaling, and opacity functions to manipulate the labels.

### Line break

You can also add a new line at any position by putting a line feed code into a string:

[PRE1]

![Line break](img/B00561_03_02.jpg)

### Text align

You can also specify the text alignment in both the horizontal and the vertical directions.

| Text alignment type | Description |
| --- | --- |
| `TextHAlignment::LEFT` | Aligns text horizontally to the left. This is the default value for horizontal alignment. |
| `TextHAlignment::CENTER` | Aligns text horizontally to the center. |
| `TextHAlignment::RIGHT` | Aligns text horizontally to the right. |
| `TextVAlignment::TOP` | Aligns text vertically to the top. This is the default value for vertical alignment. |
| `TextVAlignment::CENTER` | Aligns text vertically to the center. |
| `TextVAlignment::BOTTOM` | Aligns text vertically to the bottom. |

The following code is used for aligning text horizontally to the center:

[PRE2]

![Text align](img/B00561_03_03.jpg)

## There's more...

You can also update the string after creating the label. If you want to update the string once every second, you can do so by setting the timer as follows:

First, edit `HelloWorld.h` as follows:

[PRE3]

Next, edit `HelloWorld.cpp` as follows:

[PRE4]

First, you have to define an integer variable in the header file. Second, you need to create a label and add it on the layer. Next, you need to set the scheduler to execute the function every second. Then you can update the string by using the `setString` method.

### Tip

You can convert an int or float value to a string value by using the `StringUtils::toString` method.

A scheduler can execute the method at a specified interval. We will explain how the scheduler works in [Chapter 4](ch04.html "Chapter 4. Building Scenes and Layers"), *Building Scenes and Layers*. Refer to it for more details on the scheduler.

# Creating true type font labels

In this recipe, we will explain how to create a label with true type fonts. True type fonts are fonts that you can install into your project. Cocos2d-x's project already has two true type fonts, namely `arial.ttf` and `Maker Felt.ttf`, which are present in the `Resources/fonts` folder.

## How to do it...

Here's how to create a label by specifying a true type font. The following code can be used for creating a single-line label by using a true type font:

[PRE5]

![How to do it...](img/B00561_03_04.jpg)

## How it works...

You can create a `Label` with a true type font by specifying a label string, the path to the true type font, and the font size. The true type fonts are located in the `font` folder of `Resources`. Cocos2d-x has two true type fonts, namely `arial.ttf` and `Marker Felt.ttf`. You can generate `Label` objects of different font sizes from one true type font file. If you want to add a true type font, you can use a original true type font if you added it into the `font` folder. However, a true type font is slower than a bitmap font with respect to rendering, and changing properties such as the font face and size is an expensive operation. You have to be careful to not update it frequently.

## There's more...

If you want to create a lot of `Label` objects that have the same properties from a true type font, you can create them by specifying `TTFConfig`. `TTFConfig` has properties that are required by a true type font. You can create a label by using `TTFConfig` as follows:

[PRE6]

A `TTFConfig` object allows you to set some labels that have the same properties.

If you want to change the color of `Label`, you can change its color property. For instance, by using the following code, you can change the color to `RED`:

[PRE7]

## See also

*   You can set effects to labels. Please check the last recipe in this chapter.

# Creating bitmap font labels

Lastly, we will explain how to create a label with bitmap type fonts. Bitmap fonts are also fonts that you can install into your project. A bitmap font is essentially an image file that contains a bunch of characters and a control file that details the size and location of each character within the image. If you use bitmap fonts in your game, you can see that the bitmap fonts will be the same size on all devices.

## Getting ready

You have to prepare a bitmap font. You can create it by using a tool such as `GlyphDesigner`. We will explain this tool after [Chapter 10](ch10.html "Chapter 10. Improving Games with Extra Features"), *Improving Games with Extra Features*. Now, we will use a bitmap font in Cocos2d-x. It is located in the `COCOS_ROOT/tests/cpp-tests/Resources/fonts` folder. To begin with, you will have to add the files mentioned below to your `Resources/fonts` folder in your project.

*   `future-48.fnt`
*   `future-48.png`

## How to do it...

Here's how to create a label by specifying a bitmap font. The following code can be used for creating a single-line label using a bitmap font:

[PRE8]

![How to do it...](img/B00561_03_05.jpg)

## How it works...

You can create `Label` with a bitmap font by specifying a `label` string, the path to the true type font, and the font size. The characters in a bitmap font are made up of a matrix of dots. This font renders very fast but is not scalable. That's why it has a fixed font size when generated. A bitmap font requires the following two files: an .fnt file and a `.png` file.

## There's more...

Each character in `Label` is a `Sprite`. This means that each character can be rotated or scaled and has other changeable properties:

[PRE9]

![There's more...](img/B00561_03_06.jpg)

# Creating rich text

After creating `Label` objects on screen, you can create some effects such as a drop shadow and an outline on them easily without having your own custom class. The `Label` class can be used for applying the effects to these objects. However, note that not all label types support all effects.

## How to do it...

### Drop shadow

Here's how to create `Label` with a drop shadow effect:

[PRE10]

![Drop shadow](img/B00561_03_07.jpg)

### Outline

Here's how to create `Label` with an outline effect:

[PRE11]

![Outline](img/B00561_03_08.jpg)

### Glow

Here's how to create `Label` with a glow effect:

[PRE12]

![Glow](img/B00561_03_09.jpg)

## How it works...

Firstly, we generate a gray layer and change the background color to gray because otherwise we will not be able to see the shadow effect. Adding the effect to the label is very easy. You need to generate a `Label` instance and execute an effect method such as `enableShadow()`. This can be executed without arguments. The `enableOutline()` has two arguments, namely the outline color and the outline size. The outline size has a default value of -1\. If it has a negative value, the outline does not show. Next, you have to set the second argument. The `enableGlow` method has only one argument, namely glow color.

Not all label types support all effects, but all label types support the drop shadow effect. The `Outline` and `Glow` effects are true type font effects only. In previous versions, we had to create our own custom fonts class if we wanted to apply effects to labels. However, the current version of Cocos2d-x, version 3, supports label effects such as drop shadow, outline, and glow.

## There's more...

You can also change the shadow color and the offset. The first argument is shadow color, the second argument is the offset, and third argument is the blur radius. However, unfortunately, changing the blur radius is not supported in Cocos2d-x version 3.4.

[PRE13]

It is also possible to set two or more of these effects at the same time. The following code can be used for setting the shadow and outline effects for a label:

[PRE14]

![There's more...](img/B00561_03_10.jpg)