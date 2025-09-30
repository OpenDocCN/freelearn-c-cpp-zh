# Chapter 2. Creating Sprites

In this chapter we're going to create sprites, animations, and actions. The following topics will be covered in this chapter:

*   Creating sprites
*   Getting the sprite's position and size
*   Manipulating sprites
*   Creating animations
*   Creating actions
*   Controlling actions
*   Calling functions with actions
*   Easing actions
*   Using a texture atlas
*   Using a batch node
*   Using 3D models
*   Detecting collisions
*   Drawing a shape

# Introduction

Sprites are a 2D image. We can animate and transform them by changing their properties. Sprites are basically, items and your game is not complete without them. Sprites are not only displayed, but also transformed or moved. In this chapter, you will learn how to create sprites using 3D models in Cocos2d-x, and then, we will go through the advantages of sprites.

# Creating sprites

Sprites are the most important things in games. They are images that are displayed on the screen. In this recipe, you will learn how to create a sprite and display it.

## Getting ready

You can add the image that you made in the previous chapter into your project, by performing the following steps:

1.  Copy the image into the `Resource` folder `MyGame/Resources/res`.
2.  Open your project in Xcode.
3.  Go to **Product** | **Clean** from the Xcode menu.

You have to clean and build when you add new images into the `resource` folder. If you did not clean after adding new images, then Xcode will not recognize them. Finally, after you add the `run_01.png` to your project, your project will be seen looking like the following screenshot:

![Getting ready](img/B00561_02_01.jpg)

## How to do it...

We begin with modifying the `HelloWorld::init` method in the following code:

[PRE0]

And then, after we build & run the project, we can see the following:

![How to do it...](img/B00561_02_02.jpg)

## How it works...

You can get the screen size from the `Director::getWinSize` method. The `Director` class is a singleton class. You can get the instance using the `getInstance` method. So you can get the screen size by `Director::getInstance->getWinSize()`.

### Tip

Please note that you can get an instance of a singleton class in Cocos2d-x using the `getInstance` method.

Sprites are made from images. You can create a sprite by specifying the image. In this case, you create the sprite by `run_01.png` in the `res` folder.

Next, you need to specify the coordinates of the sprite. In this case, you set the sprite in the center of the screen. The `Size` class has the width and height property. You can specify the location of the sprite using the `setPosition` method. The argument of the `setPosition` method is `Vec2`. `Vec2` has two properties as floating point vector, `x` axis coordinate and `y` axis coordinate.

The last step is to add the sprite on the layer. A layer is like a transparent sheet on the screen. You will learn about layers in [Chapter 4](ch04.html "Chapter 4. Building Scenes and Layers"), *Building Scenes and Layers*.

All objects that are displayed on the screen are **node**. Sprite and Layer are types of node. If you haven't added it in the other nodes, the node does not appear on the screen. You can add a node in the other nodes by the `addChild` method.

## There's more...

You can set the sprite using the static coordinate. In the following case we see that the Sprite position is `(100, 200)`.

[PRE1]

Also, you can set the sprite in the center of the screen using C++ operator overloading.

[PRE2]

If you want to remove the sprite from the layer, you can remove it by the following code:

[PRE3]

## See also

The `Sprite` class has a lot of properties. You can manipulate them and change the sprite's appearance. You will also learn more about layer and the scene, which will be explained in [Chapter 4](ch04.html "Chapter 4. Building Scenes and Layers"), *Building Scenes and Layers*.

# Getting the sprite's position and size

There is a certain size and position of the sprite. In this recipe, we explain how to view the size and position of the sprite.

## How to do it...

To get the sprite position, use the following code:

[PRE4]

To get the sprite size, use the following code:

[PRE5]

## How it works...

By default, the sprite position is (`0,0`). You can change the sprite position using the `setPosition` method and get it using the `getPosition` method. You can get the sprite size using the `getContentSize` method. However, you cannot change the sprite size by the `setContentSize` method. The `contentsize` is a constant value. If you want to change the sprite size, you have to change the scale of the sprite. You will learn about the scale in the next recipe.

## There's more...

### Setting anchor points

**Anchor point** is a point that you set as a way to specify what part of the sprite will be used when setting its position. The anchor point uses a bottom-left coordinate system. By default, the anchor point of all Node objects is `(0.5, 0.5)`. This means that the default anchor point is the center.

To get the anchor point at the center of the sprite, we use the following code:

[PRE6]

To get the anchor point at the bottom-left of the sprite, we use the following code:

[PRE7]

To get the anchor point at the top-left of the sprite, we use the following code:

[PRE8]

To get the anchor point at the bottom-right of the sprite, we use the following code:

[PRE9]

To get the anchor point at the top-right of the sprite, we use the following code:

[PRE10]

The following image shows the various positions of the anchor point:

![Setting anchor points](img/B00561_02_03.jpg)

### Rectangle

To get the sprite rectangle, use the following code:

[PRE11]

`Rect` is the sprite rectangle that has properties such as `Size` and `Vec2`. If the scale is not equal to one, then `Size` in `Rect` will not be equal to the `size`, using `getContentSize` method. `Size` of `getContentSize` is the original image size. On the other side, `Size` in `Rect` using `getBoundingBox` is the size of appearance. For example, when you set the sprite to half scale, the `Size` in `Rect` using `getBoundingBox` is half the size, and the `Size` using `getContentSize` is the original size. The position and size of a sprite is a very important point when you need to specify the sprites on the screen.

## See also

*   The *Detecting collisions* recipe, where you can detect collision using `rect`.

# Manipulating sprites

A Sprite is a 2D image that can be animated or transformed by changing its properties, including its rotation, position, scale, color, and so on. After creating a sprite you can obtain access to the variety of properties it has, which can be manipulated.

## How to do it...

### Rotate

You can change the sprite's rotation to positive or negative degrees.

[PRE12]

You can get the rotation value using `getRotation` method.

[PRE13]

The positive value rotates it clockwise, and the negative value rotates it counter clockwise. The default value is zero. The preceding code rotates the sprite 30 degrees clockwise, as shown in the following screenshot:

![Rotate](img/B00561_02_04.jpg)

### Scale

You can change the sprite's scale. The default value is `1.0f`, the original size. The following code will scale to half size.

[PRE14]

You can also change the width and height separately. The following code will scale to half the width only.

[PRE15]

The following will scale to half the height only.

[PRE16]

The following code will scale that width to double and the height to half.

[PRE17]

![Scale](img/B00561_02_05.jpg)

### Skew

You can change the sprite's skew, either by `X`, `Y` or uniformly for both `X` and `Y`. The default value is zero for both `X` and `Y`.

The following code adjusts the `X` skew by `20.0`:

[PRE18]

The following code adjusts the `Y` skew by `20.0`:

[PRE19]

![Skew](img/B00561_02_06.jpg)

### Color

You can change the sprite's color by passing in a `Color3B` object. `Color3B` has an RGB value.

[PRE20]

![Color](img/B00561_02_07.jpg)

### Opacity

You can change the sprite's opacity. The opacity property is set between a value from 0 to 255.

[PRE21]

The sprite is fully opaque when it is set to 255, and fully transparent when it is set to zero. The default value is always 255.

![Opacity](img/B00561_02_08.jpg)

### Visibility

You can change the sprite's visibility by passing in a Boolean value. If it is `false`, then the sprite is invisible; if it is `true`, then the sprite is visible. The default value is always `true`.

[PRE22]

### Tip

If you want to check the sprite's visibility, use the `isVisible` method rather than the `getVisible` method. The sprite class does not have the `getVisible` method.

[PRE23]

## How it works...

A Sprite has a lot of properties. You can manipulate a sprite using the `setter` and `getter` methods.

RGB color is a 3 byte value from zero to 255\. Cocos2d-x provides predefined colors.

[PRE24]

You can find them by looking at the `ccType.h` file in Cocos2d-x.

# Creating animations

When the characters in a game start to move, the game will come alive. There are many ways to make animated characters. In this recipe, we will animate a character by using multiple images.

## Getting ready

You can create an animation from a series of the following image files:

![Getting ready](img/B00561_02_09.jpg)

You need to add the running girl's animation image files to your project and clean your project.

### Tip

Please check the recipe *Creating sprites*, which is the first recipe in this chapter, on how to add images to your project.

## How to do it...

You can create an animation using a series of images. The following code creates the running girl's animation.

[PRE25]

## How it works...

You can create an animation using the `Animation` class and the `Animate` class. They change multiple images at regular intervals. The names of the series image files have the serial number, we have added a file name to the `Animation` class in the for loop. We can create the formatted string using the `StringUtils` class in Cocos2d-x.

### Tip

`StringUtils` is a very useful class. The `StringUtils::toString` method can generate the `std::string` value from a variety of values.

[PRE26]

`StringUtils::format` method can generate the `std::string` value using the `printf` format.

You can view the log by using CCLOG macro. CCLOG is very useful. You can check the value of the variable in the log during the execution of your game. CCLOG has the same parameters as a `sprintf` function.

We will add the file name into the `Animation` instance using the `addSpriteFrameWithFile` method. It sets the units of time which the frame takes using `setDelayPerunit` method. It is set to restore the original frame when the animation finishes using the `setRestoreOriginalFrame` method. True value is to restore the original frame. It is set to the number of times the animation is going to loop. Then, create the `Animate` instance by passing it with the `Animation` instance that you created earlier. Finally, run the `runAction` method by passing in the `Animate` instance.

If you want to run the animation forever, set `-1` using the `setLoops` method.

[PRE27]

## There's more...

In the preceding code, you cannot control each animation frame. In such cases, you can use the `AnimationFrame` class. This class can control each animation frame. You can set the units of time the frame takes using the second argument of the `AnimationFrame::create` method.

[PRE28]

## See also

*   The *Using a texture atlas* recipe to create an animation using texture atlas

# Creating actions

Cocos2d-x has a lot of actions, for example, move, jump, rotate, and so on. We often use these actions in our games. This is similar to an animation, when the characters in a game start their action, the game will come alive. In this recipe you will learn how to use a lot of actions.

## How to do it...

Actions are very important effects in a game. Cocos2d-x allows you to use various actions.

### Move

To move a sprite by a specified point over two seconds, you can use the following command:

[PRE29]

To move a sprite to a specified point over two seconds, you can use the following command:

[PRE30]

### Scale

To uniformly scale a sprite by 3x over two seconds, use the following command:

[PRE31]

To scale the `X` axis by 5x, and `Y` axis by 3x over two seconds, use the following command:

[PRE32]

To uniformly scale a sprite to 3x over two seconds, use the following command:

[PRE33]

To scale `X` axis to 5x, and `Y` axis to 3x over two seconds, use the following command:

[PRE34]

### Jump

To make a sprite jump by a specified point three times over two seconds, use the following command:

[PRE35]

To make a sprite jump to a specified point three times over two seconds, use the following command:

[PRE36]

### Rotate

To rotate a sprite clockwise by 40 degrees over two seconds, use the following command:

[PRE37]

To rotate a sprite counterclockwise by 40 degrees over two seconds, use the following command:

[PRE38]

### Blink

To make a sprite blink five times in two seconds, use the following command:

[PRE39]

### Fade

To fade in a sprite in two seconds, use the following command:

[PRE40]

To fade out a sprite in two seconds, use the following command:

[PRE41]

### Skew

The following code skews a sprite's `X` axis by 45 degrees and `Y` axis by 30 degrees over two seconds:

[PRE42]

The following code skews a sprite's `X` axis to 45 degrees and `Y` axis to 30 degrees over two seconds.

[PRE43]

### Tint

The following code tints a sprite by the specified RGB values:

[PRE44]

The following code tints a sprite to the specified RGB values:

[PRE45]

## How it works...

`Action` objects make a sprite perform a change to its properties. `MoveTo`, `MoveBy`, `ScaleTo`, `ScaleBy` and others, are `Action` objects. You can move a sprite from one position to another position using `MoveTo` or `MoveBy`.

You will notice that each `Action` has a `By` and `To` suffix. That's why they have different behaviors. The method with the `By` suffix is relative to the current state of sprites. The method with the `To` suffix is absolute to the current state of sprites. You know that all actions in Cocos2d-x have `By` and `To` suffix, and all actions have the same rule as its suffix.

## There's more...

When you want to execute a sprite action, you make an action and execute the `runAction` method by passing in the `action` instance. If you want to stop the action while sprites are running actions, execute the `stopAllActions` method or `stopAction` by passing in the `action` instance that you got as the return value of the `runAction` method.

[PRE46]

If you run `stopAllActions`, all of the actions that sprite is running will be stopped. If you run `stopAction` by passing the `action` instance, that specific action will be stopped.

# Controlling actions

In the previous recipe, you learned some of the basic actions. However, you may want to use more complex actions; for example, rotating a character while moving, or moving a character after jumping. In this recipe, you will learn how to control actions.

## How to do it...

### Sequencing actions

`Sequence` is a series of actions to be executed sequentially. This can be any number of actions.

[PRE47]

The preceding command will execute the following actions sequentially:

*   Move a sprite 100px to the right over two seconds
*   Rotate a sprite clockwise by 360 degree over two seconds

It takes a total of four seconds to execute these commands.

### Spawning actions

`Spawn` is very similar to `Sequence`, except that all actions will run at the same time. You can specify any number of actions at the same time.

[PRE48]

It will execute the following actions at the same time:

*   Moved a sprite 100px to the right over two seconds
*   Rotated a sprite clockwise by 360 degree over two seconds

It takes a total of two seconds to execute them.

### Repeating actions

`Repeat` object is to repeat an action the number of specified times.

[PRE49]

The preceding command will execute a `rotate` action five times.

If you want to repeat forever, you can use the `RepeatForever` action.

[PRE50]

### Reversing actions

If you generate an `action` instance, you can call a `reverse` method to run it in the `reverse` action.

[PRE51]

The preceding code will execute the following actions sequentially:

*   Move a sprite 100px to the right over two seconds.
*   Move a sprite 100px to the left over two seconds.

In addition, if you generate a sequence action, you can call a `reverse` method to run it in the opposite order.

[PRE52]

The preceding code will execute the following actions sequentially:

*   Move a sprite 100px to the right over two seconds.
*   Rotate a sprite clockwise by 360 degree over two seconds
*   Rotate a sprite counterclockwise by 360 degree over two seconds
*   Move a sprite 100px to the left over two seconds.

### DelayTime

`DelayTime` is a delayed action within the specified number of seconds.

[PRE53]

The preceding command will execute the following actions sequentially:

*   Move a sprite 100px to the right over two seconds
*   Delay the next action by two seconds
*   Rotate a sprite clockwise by 360 degree over two seconds

It takes a total of six seconds to execute it.

## How it works...

`Sequence` action runs actions sequentially. You can generate a `Sequence` instance with actions sequentially. Also, you need to specify `nullptr` last. If you did not specify `nullptr`, your game will crash.

![How it works...](img/B00561_02_10.jpg)

`Spawn` action runs actions at the same time. You can generate a `Spawn` instance with actions and `nullptr` like `Sequence` action.

![How it works...](img/B00561_02_11.jpg)

`Repeat` and `RepeatForever` actions can run, repeating the same action. `Repeat` action has two parameters, the repeating action and the number of repeating actions. `RepeatForever` action has one parameter, the repeating action, which is why it will run forever.

Most actions, including `Sequence`, `Spawn` and `Repeat,` have the `reverse` method. But like the `MoveTo` method that has the suffix `To`, it does not have the `reverse` method; that's why it cannot run the reverse action. `Reverse` method generates its reverse action. The following code uses the `MoveBy::reverse` method.

[PRE54]

`DelayTime` action can delay an action after this. The benefit of the `DelayTime` action is that you can put it in the `Sequence` action. Combining `DelayTime` and `Sequence` is a very powerful feature.

## There's more...

`Spawn` produces the same results as running multiple consecutive `runAction` statements.

[PRE55]

However, the benefit of `Spawn` is that you can put it in the `Sequence` action. Combining `Spawn` and `Sequence` is a very powerful feature.

[PRE56]

![There's more...](img/B00561_02_12.jpg)

# Calling functions with actions

You may want to call a function by triggering some actions. For example, you are controlling the sequence action, jump, and move, and you want to use a sound for the jumping action. In this case, you can call a function by triggering this jump action. In this recipe, you will learn how to call a function with actions.

## How to do it...

Cocos2d-x has the `CallFunc` object that allows you to create a function and pass it to be run in your `Sequence`. This allows you to add your own functionality to your `Sequence` action.

[PRE57]

The preceding command will execute the following actions sequentially:

*   Move a sprite 100px to the right over two seconds
*   Rotate a sprite clockwise by 360 degrees over two seconds
*   Execute CCLOG

## How it works...

The `CallFunc` action is usually used as a callback function. For example, if you want to perform a different process after finishing the `move` action. Using `CallFunc`, you can call the method at any time. You can use a lambda expression as a callback function.

If you get a callback with parameters, its code is the following:

[PRE58]

The instance of this parameter is the sprite that is running the action. You can get the sprite instance by casting to `Sprite` class.

Then, you can also specify a callback method. `CallFunc` has `CC_CALLBACK_0` macro as an argument. `CC_CALLBACK_0` is a macro for calling a method without parameters. If you want to call a method with one parameter, you need to use the `CallFuncN` action and `CC_CALLBACK_1` macro. `CC_CALLBACK_1` is a macro for calling a method with one argument. A parameter of a method that is called by `CallFuncN` is the `Ref` class.

You can call a method using the following code:

[PRE59]

To call a method with an argument, you can use the following code:

[PRE60]

## There's more...

To combine the `CallFuncN` and the `Reverse` action, use the following code:

[PRE61]

The preceding command will execute the following actions sequentially:

*   Move a sprite 100px to the right over two seconds
*   Rotate a sprite clockwise by 360 degree over two seconds
*   Move a sprite 100px to the left over two seconds

# Easing actions

**Easing** is animating with a specified acceleration to make the animations smooth. Ease actions are a good way to fake physics in your game. If you use easing actions with your animations, your game looks more natural with smoother animations.

## How to do it...

Let's move a `Sprite` object from `(200,200)` to `(500,200)` with acceleration and deceleration.

[PRE62]

Next, let's drop a `Sprite` object from the top of the screen and make it bounce.

[PRE63]

## How it works...

The animation's duration time is the same time regardless of whether you use easing. `EaseIn`, `EaseOut` and `EaseInOut` have two parameters—the first parameter is the action by easing, the second parameter is rate of easing. If you specified this parameter to `1.0f`, this easing action is the same without easing. Anything over `1.0f`, means easing is fast, under `1.0f`, and easing will be slow.

The following table are typical easing types.

| Class Name | Description |
| --- | --- |
| `EaseIn` | Moves while accelerating. |
| `EaseOut` | Moves while decelerating. |
| `EaseInOut` | Start moving while accelerating, stop while decelerating. |
| `EaseExponentialIn` | It's similar to `EaseIn`, but meant to accelerate at a rate of exponential curve. It is also used with `Out` and `InOut` like `EaseIn`. |
| `EaseSineIn` | It's similar to `EaseIn`, but meant to accelerate at a rate of sin curve. It is also used with `Out` and `InOut` like `EaseIn`. |
| `EaseElasticIn` | Moves after shaking slowly, little by little. It is also used with `Out` and `InOut` like `EaseIn`. |
| `EaseBounceIn` | Moves after bouncing. It is also used with `Out` and `InOut` like `EaseIn`. |
| `EaseBackIn` | Moves after moving in the opposite direction. It is also used with `Out` and `InOut` like `EaseIn` |

This is a graph that displays typical easing functions:

![How it works...](img/B00561_02_13.jpg)

# Using a texture atlas

A **texture atlas** is a large image containing a collection of each sprite. We often use a texture atlas rather than individual images. In this recipe, you will learn how to use a texture atlas.

## Getting ready

You have to add the texture atlas files into your project and clean your project.

*   `running.plist`
*   `running.png`

## How to do it...

Let's try to read the texture altas file and make a sprite from it.

[PRE64]

## How it works...

Firstly, we loaded the texture atlas file, when the `SpritFrameCache` class cached all the images that are included in it. Secondly, you generated a sprite. Do not use the `Sprite::create` method to generate it, use the `Sprite::createWithSpriteFrameName` method instead. Then, you can handle the sprite as a normal sprite.

A texture atlas is a large image containing a collection of images. It is composed of a `plist` file and a `texture` file. You can create a texture atlas by using tools. You will learn how to make a texture atlas using tools in [Chapter 10](ch10.html "Chapter 10. Improving Games with Extra Features"), *Improving Games with Extra Features*. A `plist` file is defined as the original file name of the image and it is located within the `texture` file. It also defines the image that will be used by the texture atlas. The `plist` file for the texture atlas is xml format as follows.

[PRE65]

![How it works...](img/B00561_02_14.jpg)

Why would we use the texture atlas? Because using the memory efficiently is good. Double the memory size is required when the computer loads the image into the memory. For example, there are ten images that are 100x100 size. We will use nine images, but one image requires memories for 128x128 size. On the other hand, texture atlas is one image containing a collection of nine images, where the image size is 1000x1000\. It requires a memory size of 1024x1024\. This is why texture atlas is used to save wasting unnecessary memory usage.

## There's more...

The size of the texture altas can vary in usage depending on the devices. You can check the maximum texture size of the device in the following codes:

[PRE66]

You can generate an animation using a texture atlas and a `plist` file. Firstly, you have to add `run_animation.plist` file into your project. The file is shown in the following screenshot:

![There's more...](img/B00561_02_15.jpg)

This `plist` defines a frame animation. In this case, we defined an animation called `run` using images from `run_01.png` to `run_08.png`. And the animation will loop forever if you specify `-1` to `loop` key's value. The texture atlas was specified `running.plist`.

Secondly, you need to generate an animation using the `plist` file.

[PRE67]

You also need to cache animation data using the `AnimationCache::addAnimationWithFile` method with the animation `plist`. Next, you will generate an `Animation` instance by specifying `run` that was defined as an animation name in the `plist`. And then, you generate an action from the animation. After that, you can animate using `runAction` method with the action instance.

## See also

It is very difficult to create a texture atlas manually. You had better use a tool such as the `TexturePacker`, which you will learn about in [Chapter 11](ch11.html "Chapter 11. Taking Advantages"), *Taking Advantages*.

# Using a batch node

Renderer speed will be slow if there are a lot of sprites on the screen. However, a shooting game needs a lot of images such as bullets, and so on. In this time, if renderer speed is slow, the game earns a bad review. In this chapter, you will learn how to control a lot of sprites.

## How to do it...

Let's try to display a lot of sprites using `SpriteBatchNode`.

[PRE68]

## How it works...

The `SpriteBatchNode` instance can be used to do the following:

*   Generate a `SpriteBatchNode` instance using a texture
*   Add the instance on the layer
*   Generate sprites using the texture in the `SpriteBatchNode` instance
*   Add these sprites on the `SpriteBatchNode` instance

`SpriteBatchNode` can reference only one texture (one image file or one texture atlas). Only the sprites that are contained in that texture can be added to the `SpriteBatchNode`. All sprites added to a `SpriteBatchNode` are drawn in one OpenGL ES draw call. If the sprites are not added to a `SpriteBatchNode` then an OpenGL ES draw call will be needed for each one, which is less efficient.

## There's more...

The following screenshot is an executing screen image. You can see three lines of information for Cocos2d-x on the left bottom corner. The top line is the number of polygon vertices. The middle line is the number of OpenGL ES draw call. You understand that a lot of sprites are drawn by one OpenGL ES draw call. The bottom line is FPS and seconds per frame.

![There's more...](img/B00561_02_16.jpg)

### Tip

If you want to hide this debug information, you should set a `false` value to the `Director::setDisplayStats` method. You will find it in the `AppDelegate.cpp` in your project.

[PRE69]

Since Cocos2d-x version 3, the `auto batch` function of draw calls has been added, Cocos2d-x can draw a lot of sprites with one OpenGL ES draw call, without `SpriteBatchNode`. However, it has the following conditions:

*   Same texture
*   Same `BlendFunc`

# Using 3D modals

Cocos2d-x version 3 supports an exciting new function called **3D modals**. We can use and display 3D modals in Cocos2d-x. In this recipe, you will learn how to use 3D modals.

## Getting ready

You have to add the 3D object data into your project and clean your project. The resource files present in the `COCOS_ROOT/test/cpp-tests/Resources/Sprite3DTest` folder are—`body.png` and `girl.c3b`

## How to do it...

Let's try to display a 3D model and move it.

[PRE70]

![How to do it...](img/B00561_02_17.jpg)

## How it works...

You can create the 3D sprite from a 3D model in the same way as we made a 2D sprite and displayed it. The `Placement` method and the `action` method is exactly the same as seen in a 2D sprite. You can create the `Animation3D` instance from the animation data that is defined in the 3D model.

## There's more...

Finally you will try to move the 3D sprite to the left or right. You will notice that 3D sprites differ in appearance depending on their position on the screen when you run the following code:

[PRE71]

## See also

You can use 3D data formats such as obj, c3b, and c3t. “c3t” stands for Cocos 3d binary. You can get this formatted data by converting `fbx` files.

# Detecting collisions

In an action game, a very important technique is to detect collisions between each sprite. However, it is pretty complicated to detect collisions between `rect` and `rect` or `rect` and `point`. In this recipe, you will learn how to detect collisions easily.

## How to do it...

There are two ways to detect collisions. The first method checks whether a point is contained within the rectangle of the sprite.

[PRE72]

The second method checks whether two sprite's rectangles have overlapped.

[PRE73]

## How it works...

The `Rect` class has two properties—`size` and `origin`. The `size` property is the sprite's size. The origin property is the sprite's left-bottom coordinate. Firstly, you get the sprite's `rect` using the `getBoundingBox` method.

![How it works...](img/B00561_02_18.jpg)

Using the `Rect::containsPoint` method by specifying the coordinate, it is possible to detect whether it contains the rectangle. If it contains it, the method returns `true`. Using `Rect::intersectsRect` method by specifying another rectangle, it is possible to detect whether they overlap. If they overlap, the method returns `true`.

The following image shows a collision between `rect` and `point` or `rect` and `rect`:

![How it works...](img/B00561_02_19.jpg)

## There's more...

The `Rect` class has more methods including `getMinX`, `getMidX`, `getMaxX`, `getMinY`, `getMidY`, `getMaxY` and `unionWithRect`. You can obtain the value in the following figure using each of these methods.

![There's more...](img/B00561_02_20.jpg)

## See also

*   If you used the physics engine, you can detect collision in a different way. Take a look at [Chapter 9](ch09.html "Chapter 9. Controlling Physics"), *Controlling Physics*.
*   If you want to detect collision with consideration of the transparent parts of an image, take a look at [Chapter 11](ch11.html "Chapter 11. Taking Advantages"), *Taking Advantages*.

# Drawing a shape

Drawing a shape in Cocos2d-x can be easy using the `DrawNode` class. If you can draw various shapes using `DrawNode`, you will to need to prepare textures for such shapes. In this section, you will learn how to draw shapes without textures.

## How to do it...

Firstly, you made a `DrawNode` instance as shown in the following codes. You got a window size as well.

[PRE74]

### Drawing a dot

You can draw a dot by specifying the point, the radius and the color.

[PRE75]

![Drawing a dot](img/B00561_02_21.jpg)

### Drawing lines

You can draw lines by specifying the starting point, the destination point, and the color. A `1px` thick line will be drawn when you use the `drawLine` method. If you want to draw thicker lines, use the `drawSegment` method with a given radius.

[PRE76]

![Drawing lines](img/B00561_02_22.jpg)

### Drawing circles

You can draw circles as shown in the following codes. The specification of the arguments is as follows:

*   center position
*   radius
*   angle
*   segments
*   draw a line to center or not
*   scale x axis
*   scale y axis
*   color

[PRE77]

![Drawing circles](img/B00561_02_23.jpg)

Segment is the number of vertices of the polygon. As you know, the circle is a polygon that has a lot of vertices. Increasing the number of vertices is close to a smooth circle, but the process load goes up. Incidentally, you should use `drawSolidCircle` method if you want to get a solid circle.

### Drawing a triangle

You can draw a triangle as in the following code with three vertices and the color.

[PRE78]

![Drawing a triangle](img/B00561_02_24.jpg)

### Drawing rectangles

You can draw rectangles using the following code with the left-bottom point, the right-top point, and the color. You can draw fill color if you use the `drawSolidRect` method.

[PRE79]

![Drawing rectangles](img/B00561_02_25.jpg)

### Drawing a polygon

You can draw a polygon using the following code with the given vertices, the number of vertices, filling color, border's width, and border's color.

[PRE80]

![Drawing a polygon](img/B00561_02_26.jpg)

### Drawing a Bezier curve

You can draw a Bezier curve as shown in the following code. Using `drawQuadBezier` method, you can draw a quadratic Bezier curve, and using `drawCubicBezier` method you can draw a cubic Bezier curve. The third argument of the `drawQuadBezier` method and the fourth argument of the `drawCubicBezier` method is the number of vertices in the same way as the circle.

[PRE81]

![Drawing a Bezier curve](img/B00561_02_27.jpg)

## How it works...

`DrawNode` is like a mechanism that enables Cocos2d-x to process at a high speed, by making drawing shapes all at once and not separately, or one by one. When you draw multiple shapes, you should use one `DrawNode` instance, instead of multiple `DrawNode` instances and then add multiple shapes in it. Also `DrawNode` does not have the concept of depth. Cocos2d-x will draw to the order of the added shapes in `DrawNode`.