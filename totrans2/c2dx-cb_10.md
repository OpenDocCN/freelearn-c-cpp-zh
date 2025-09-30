# Chapter 10. Improving Games with Extra Features

The following topics will be covered in this chapter:

*   Using Texture Packer
*   Using Tiled Map Editor
*   Getting the property of the object in the tiled map
*   Using Physics Editor
*   Using Glyph Designer

# Introduction

For a long time, there have been a lot of tools available to you that help you in game development. Some of these tools can be used in Cocos2d-x. With the use of these tools, you can quickly and efficiently develop your game. You can, for example, use original fonts and create sprite sheets, a map like a role-playing game, complex physical objects, and so on. In this chapter, you will learn how to use these extra tools in your game development.

# Using Texture Packer

**Texture Packer** is a tool that can drag and drop images and publish. With the use of this tool, we can not only create sprite sheets, but also export multi sprite sheets. If there are a lot of sprites, then we need to use the command line tool when we create sprite sheets, encrypt them, and so on. In this recipe, you can use Texture Packer.

## Getting ready

Texture Packer is a paid application. However, you can use the free trial version. If you don't have it, you can download it by visiting [https://www.codeandweb.com/texturepacker](https://www.codeandweb.com/texturepacker)

## How to do it...

1.  You need to launch Texture Packer, after which you will see a blank window appear.![How to do it...](img/B0561_10_01.jpg)
2.  In this recipe, we will use these sprites as shown in the following screenshot:![How to do it...](img/B0561_10_02.jpg)
3.  You simply need to drag the images into the Texture Packer window and it will automatically read all the files and arrange them.![How to do it...](img/B0561_10_03.jpg)
4.  And that's it. So let's publish the sprite sheet image and `plist` to click the **publish** button. That's how you can get the sprite sheet image and `plist`.

## How it works...

You can get the sprite sheet image and `plist` file. In this part, we explain how to publish the sprite sheet for all devices with a single click.

1.  Click on the **AutoSD** button with the gear icon, and you will see an additional window appear, as shown in the following screenshot:![How it works...](img/B0561_10_04.jpg)
2.  Select the **cocos2d-x HDR/HD/SD** and click the **Apply** button. After clicking it, setting the default scale, extension, size and so on like in the following image:![How it works...](img/B0561_10_05.jpg)
3.  Next, you have to click the **publish** button, you will see the window to select the data file name. The important thing is to select the folder named `HDR` as in the following image:![How it works...](img/B0561_10_06.jpg)
4.  Finally, you will get three size sprite sheets automatically as in the following image:![How it works...](img/B0561_10_07.jpg)

The sprite sheet in `HDR` folder is the largest size. The images that were dragged and dropped are HDR images. These images are good for resizing to HD or SD images.

## There's more…

You can use the Texture Packer on the command like like this:

[PRE0]

The preceding command is to make a sprite sheet named `hoge.plist` and `hoge.png` by using images named `foo_*.png`. For example, if there were `foo_1.png` to `foo_10.png` in a folder, then the sprite sheet is created from these 10 images.

In addition, the command has other options as in the following table:

| Option | Description |
| --- | --- |
| `--help` | Display help text |
| `--version` | Print version information |
| `--max-size` | Set the maximum texture size |
| `--format cocos2d` | Format to write, default is cocos2d |
| `--data` | Name of the data file to write |
| `--sheet` | Name of the sheet to write |

There are a lot of options other than that. You can see another options by using the following command:

[PRE1]

# Using Tiled Map Editor

A tiled map is a grid of cells where the value in the cell indicates what should be at the location. For example, (0,0) is a road, (0,1) is a grass, (0,2) is a river and so on. Tiled maps are very useful but they are pretty hard to create by hand. **Tiled** is a tool that can be used to just create tiled maps. Tiled is a free application. However, this application is a very powerful, useful and popular tool. There are various kinds of Tiled Map, for example, 2D maps such as Dragon Quest, Horizontal scrolling game map such as Super Mario and so on. In this recipe, you can basically use texture packer.

## Getting ready

If you don't have Tiled Map Editor, you can download it from [https://www.mapeditor.org/](https://www.mapeditor.org/).

And then, after downloading it, you will install the application and copy the `example` folder in the `dmg` file, into the working space of your computer.

Tiled Map Editor is free application. However, you can donate to this software if you like.

## How to do it...

In this part, we explain how to create a new map from scratch with the Tiled tool.

1.  Launch Tiled and selecting **File** | **New** in the menu. Open the new additional window as in the following image:![How to do it...](img/B0561_10_08.jpg)
2.  Select XML in **Tile layer format** and change **Width** and **Height** in **Map size** to 50 tiles. Finally, click **OK**. So you can see the Tiled's window as in the following image:![How to do it...](img/B0561_10_09.jpg)
3.  Select **Map** | **New Tileset…** in the menu. You can select the tileset window. Select the tileset image by clicking the **Browse…** button in the middle of the window. In this case, you will select `tmw_desert_spacing.png` file in Tiled's `example` folder. This tileset has tiles with a width and height of 32px and a margin and spacing of 1px. So you have to change these values as shown in the following screenshot:![How to do it...](img/B0561_10_10.jpg)
4.  Finally, click on the **OK** button, and you will see the new editor window as shown in the following screenshot:![How to do it...](img/B0561_10_11.jpg)
5.  Next, let's try to paint the ground layer using the tile that you selected. Select the tile from the right and lower panes, and select the bucket icon in the tool bar. Then, click on the map, and you will see the ground painted with the same tile.![How to do it...](img/B0561_10_12.jpg)
6.  You can arrange the tiles on the map. Select the tile in the lower right pane and select the stamp icon in the tool bar. Then, click on the map. That's how you can put the tile on the map.![How to do it...](img/B0561_10_13.jpg)
7.  After you have finished arranging the map, you need to save it as a new file. Go to **File** | **Save as…** in the menu and save the new file that you made. To use Cococs2d-x, you have to add the `tmx` file and tileset image file into the `Resources/res` folder in your project. In this recipe, we added `desert.tmx` and `tmw_desert_spacing.png` in Tiled's `example` folder.![How to do it...](img/B0561_10_14.jpg)
8.  From now on, you have to work in Xcode. Edit the `HelloWorld::init` method as shown in the following code:

    [PRE2]

9.  After building and running, you can see the following image on the simulator or devices:

![How to do it...](img/B0561_10_15.jpg)

## How it works...

The files that Tiled map needs are the `tmx` file and tileset image file. That's why you have to add these files into your project. You can see the Tiled map object using the `TMXTiledMap` class. You have to specify the `tmx` file path to the `TMXTiledMap::create` method. The `TMXTiledMap` object is Node. You can see the tiled map only when you add the `TMXTiledMap` object using the `addChild` method.

[PRE3]

### Tip

`TMXTileMap` object's anchor position is `Vec2(0,0)`. The normal node's anchor position is `Vec2(0.5f, 0.5f)`.

## There's more…

The tiled map is huge. So, we try to move the map by scrolling it. In this case, you touch the screen and scroll the map by the distance from the touching point to the center of the screen.

1.  Add the following code in the `HelloWorld::init` method:

    [PRE4]

2.  Define the `touch` method and some properties in `HelloWorldScene.h` as shown in the following code:

    [PRE5]

3.  Add the touch method in `HelloWorldScene.cpp` as shown in the following code:

    [PRE6]

4.  Finally, add the `update` method in `HelloWorldScene.cpp` as shown in the following code:

    [PRE7]

After that, run this project and touch the screen. This is how you can move the map in the direction that you swipe.

# Getting the property of the object in the tiled map

Now, you can move the Tiled map. However, you might notice the object on the map. For example, if there is a wood or wall in the direction of movement, you can't move in that direction beyond that object. In this recipe, you will notice the object on the map by getting the property of it.

## Getting ready

In this recipe, you will make a new property of the tree object and set a value to it.

1.  Launch the Tiled application and reopen the `desert.tmx` file.
2.  Select the tree object in the **Tilesets** window.
3.  Add a new property by clicking on the plus icon in the lower left corner in the **Properties** window. Then, a window will pop up specifying the property's name. Enter `isTree` in the text area.
4.  After you name the new property, it will be shown in the properties list. However, you will find that its value is empty. So, you have to set the new value to it. In this case, you need to set a true value as shown in the following image:![Getting ready](img/B0561_10_16.jpg)
5.  Save it and update `desert.tmx` in your project.

## How to do it...

In this recipe, you will get the property of the object that you touched.

1.  Edit the `HelloWorld::init` method to show the tiled map and add the event listener for touching.

    [PRE8]

2.  Add the `HelloWorld::getTilePosition` method. You can get the tile's grid row/column position if you called this method by specifying the touch position.

    [PRE9]

3.  Finally, you can get the properties of the object that you touch. Add the `HelloWorld::onTouchBegan` method as shown in the following code:

    [PRE10]

Let's build and run this project. If you touched the tree to which you set the new `isTree` property, you can see *it's tree!* in the log.

## How it works...

There are two points in this recipe. The first point is getting the tile's row/column position on the tiled map. The second point is getting the properties of the object on the tiled map.

Firstly, let's explain how to get the tiles' row/column position on the tiled map.

1.  Get the map size using the `TMXTiledMap::getContentSize` method.

    [PRE11]

2.  Calculate the `point` on the map from the touching point and map position.

    [PRE12]

3.  Get the tile size using the `TMXTiledMap::getTileSize` method.

    [PRE13]

4.  Get the row/column of the tile in the map using the `TMXTiledMap::getMapSize` method.

    [PRE14]

5.  Get the magnification display using the original size called `mapContentSize` and real size calculated by the column's width and tile's width.

    [PRE15]

6.  The origin of coordinates for the tiles is located in the upper left corner. That's why the tile's row/column position of the tile that you touched is calculated using the tile's size, the row, and magnification display as shown in the following code:

    [PRE16]

    `tilePoint.x` is the column position and `tilePoint.y` is row position.

Next, let's take a look at how to get the properties of the object on the Tiled map.

1.  Get the row/column position of the tile that you touched using the touching point.

    [PRE17]

2.  Get the layer called `"Ground"` from the tiled map.

    [PRE18]

3.  There are the objects on this layer called `Ground`. Get the `TileGID` from this layer using row/column of the tile.

    [PRE19]

4.  Finally, get the properties as `ValueMap` from the map using the `TMXTiledMap::getPropertiesForGID` method. Then, get the `isTree` property's value from them as shown in the following code:

    [PRE20]

In this recipe, we showed only the log. However, in your real game, you will add the point to the object, explosions and so on.

# Using Physics Editor

In [Chapter 9](ch09.html "Chapter 9. Controlling Physics"), *Controlling Physics*, you learned about **Physics Engine**. We can create physics bodies to use Cocos2d-x API. However, we can only create a circle shape or a box shape. Actually, you have to use complex shapes in real games. In this recipe, you will learn how to create a lot of shapes using **Physics Editor**.

## Getting ready

Physics Editor is created by the same company that created Texture Packer. Physics Editor is a paid application. But you can use a free trial version. If you don't have it, you can download it by visiting the [https://www.codeandweb.com/physicseditor](https://www.codeandweb.com/physicseditor)

Here, you prepare the image to use this tool. Here, we will use the following image that is similar to a gear. This image's name is `gear.png`.

![Getting ready](img/B0561_10_17.jpg)

## How to do it...

First of all, you will create a physics file to use Physics Editor.

1.  Launch Physics Editor. Then, drag the image `gear.png` to the left pane.![How to do it...](img/B0561_10_18.jpg)
2.  Click on the shaper tracer icon that is the third icon from the left in the tool bar. The shaper tracer icon is shown in the following image:![How to do it...](img/B0561_10_a1.jpg)
3.  After this, you can see the pop-up window as shown in the following image:![How to do it...](img/B0561_10_19.jpg)

    You can change the **Tolerance** value. If the **Vertexes** value is too big, the renderer is slow. So you set the suitable **Vertexes** value to change the **Tolerance** value. Finally, click on the **OK** button. You will see the following:

    ![How to do it...](img/B0561_10_20.jpg)
4.  Select `Cocos2d-x` in **Exporter**. In this tool, the anchor point's default value is `Vec2(0,0)`. In Cocos2d-x, the anchor point's default is `Vec2(0.5f, 0.5f)`. So you should change the anchor point to the center as shown in the following screenshot:![How to do it...](img/B0561_10_21.jpg)
5.  Check the checkboxes for **Category**, **Collision**, and **Contact**. You need to scroll down to see this window in the right pane. You can check all the checkboxes and click all buttons that are in the bottom of the right pane.
6.  Publish the `plist` file to use this shape in Cocos2d-x. Click on the **Publish** button and save as the previous name.
7.  You can see the **Download loader code** link under the **Exporter** selector. Click on the link. After this, open the browser and browse to the github page. Cocos2d-x cannot load Physics Editor's `plist`. However, the loader code is provided in github. So you have to clone this project and add the codes in the `Cocos2d-x` folder in the project.![How to do it...](img/B0561_10_22.jpg)

Next, you will write code to create the physics bodies by using the Physics Editor data. In this case, the gear object will appear at the touching point.

1.  Include the file `PhysicsShapeCache.h`.

    [PRE21]

2.  Create a scene with the physics world as shown in the following code:

    [PRE22]

3.  Create a wall of the same screen size in the scene and add the touching event listener. Then, load the Physics Editor's data as shown in the following code:

    [PRE23]

4.  Make the gear objects perform when touching the screen as shown in the following code:

    [PRE24]

5.  After this, build and run this project. After touching the screen, the gear objects appear at the touching point.![How to do it...](img/B0561_10_23.jpg)

## How it works...

1.  Firstly, you have to add two files, `plist` and image. Physics body is defined in the `plist` file that you published with Physics Editor. However, you use the gear image to create a sprite. Therefore, you have to add the `plist` file and `gear.png` into your project.
2.  Cocos2d-x cannot read Physics Editor's data. Therefore, you have to add the loader class that is provided in github.
3.  To use the Physics Engine, you have to create a scene with Physics World and you should set the debug draw mode to easy, to better understand physics bodies.

    [PRE25]

4.  Without border or walls, the physics objects will drop out of the screen. So you have to put up a wall that is the same size as the screen.

    [PRE26]

5.  Load the physics data's `plist` that was created by Physics Editor. The `PhysicsShapeCache` will load the `plist` at once. After that, the physics data is cached in the `PhysicsShapeCache` class.

    [PRE27]

6.  In the `HelloWorld::onTouchBegan` method, create the gear object at the touching point. You can create physics body using the `PhysicsShapeCache::createBodyWithName` method with physics object data.

    [PRE28]

# Using Glyph Designer

In games, you have to use text frequently. In which case, if you used the system font to display the text, you will have some problems. That's why there are different fonts for each device. The bitmap fonts are faster to render than the TTF fonts. So, Cocos2d-x uses the bitmap font to display the fps information in the bottom-left corner. Therefore, you should add the bitmap font into your game to display the text. In this recipe, you will learn how to use **Glyph Designer** which is the tool to make the original bitmap font and how to use the bitmap font in Cocos2d-x.

## Getting ready

Glyph Designer is a paid application. But you can use a free trial version. If you don't have it, you can download it by visiting the following URL:

[https://71squared.com/glyphdesigner](https://71squared.com/glyphdesigner)

Next, we will find a free font that fits your game's atmosphere. In this case, we will use the font called `Arcade` from the dafont site ([http://www.dafont.com/arcade-ya.font](http://www.dafont.com/arcade-ya.font)). After downloading it, you need to install it to your computer.

On the dafont site, there are a lot of fonts. However, the font license is different for each font. If you used the font, you need to check its license.

## How to do it...

In this section, you will learn how to use Glyph Designer.

1.  Launch Glyph Designer. In the left pane, there are all the fonts that are installed on your computer. You can choose the font that you want to use in your game from there. Here we will use `Arcade` font that you downloaded a short time ago. If you didn't install it yet, you can load it. To load the font, you have to click on the **Load Font** button in the tool bar.![How to do it...](img/B0561_10_24.jpg)
2.  After selecting or loading the font, it is displayed in the center pane. If your game used a part of the font, you have to hold the characters that you need to save memory and the application capacity. To select the characters, you can use the **Include Glyphs** window in the right pane. You need to scroll down to see this window in the right pane.![How to do it...](img/B0561_10_25.jpg)
3.  The others, you can specify the size, color, and shadow. In the **font color** option, you can set a gradient.
4.  Finally, you can create an original font by clicking on the `Export` icon on the right side of the tool bar.
5.  After exporting, you will have the two files that have the extension of `.fnt` and `.png`.

## How it works...

The bitmap font has two files, `.fnt` and `.png`. These files are paired for use in the bitmap font. Now, you will learn how to use bitmap fonts in Cocos2d-x.

1.  You have to add the font that were created in Glyph Designer, into the `Resources/font` folder in your project.
2.  Add the following code to display "`Cocos2d-x`" in your game.

    [PRE29]

3.  After building and running your project, you will see the following:![How it works...](img/B0561_10_26.jpg)

## There's more…

Some fonts aren't monospaced. The true type font is good enough for use in a word-processor. However, the monospaced font is more attractive. For example, the point character needs to use the monospaced font. When you want to make the monospaced font into a non-monospaced font, you can go through the following steps:

1.  Check the checkbox named **Fixed Width** in **Texture Atlas** in right pane.
2.  Preview your font and click on the **Preview** icon in the tool bar. Then, you can check the characters that you want to check in the textbox.
3.  If you want to change the character spacing, then you need to change the number next to the checkbox of **Fixed Width**.![There's more…](img/B0561_10_27.jpg)