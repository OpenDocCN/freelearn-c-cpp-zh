# Chapter 5. Creating GUIs

In this chapter, we're going to create various UI parts. The following topics will be covered in this chapter:

*   Creating menus
*   Creating buttons
*   Creating checkboxes
*   Creating loading bar
*   Creating sliders
*   Creating text fields
*   Creating scroll views
*   Creating page views
*   Creating list views

# Introduction

Games have a lot of GUI parts, for example, there are menus, buttons, checkboxes, loading bars, and so on. We cannot make our game without these parts. Further, these are a little different from the node we've discussed until now. In this chapter we will see how to create various GUI parts such as menus, sliders, text fields etc. for a game.

# Creating menus

In this recipe, we will create a menu. A menu has various buttons such as a start button and a pause button. A Menu is a very important component for any game and they are really useful too. The steps to use a menu are little complex. In this recipe, we will have a glance over creating menus to understand its complexity and to get used to them.

## Getting ready

We prepared the following image as a button image and added them to the `Resources/res` folder in our project. We will use the following image of the button to use it as menu:

![Getting ready](img/0561_05_01.jpg)

## How to do it...

Firstly, we will create a simple menu that has one item for a button. We will use the `item1.png` file as the button image. Create the menu by using the code here.

[PRE0]

The execution result of this code is shown in the following image:

![How to do it...](img/0561_05_02.jpg)

Further, you can see the `tapped item` text in the log after tapping the menu item. You will notice that the button becomes a little dark when you tap it.

## How it works...

1.  Create a sprite of the normal status when the button is not operated.
2.  Create a sprite of the selected status when the button is pressed. In this case, we used the same images for both the normal status and the selected status, but players could not understand the change in status when they tapped the button. That's why we changed the selected image to a slightly darker image by using the `setColor` method.
3.  Create an instance of the `MenuItemSprite` class by using these two sprites. The third argument specifies the lambda expression to be processed when the button is pressed.

This time, we created only one button in the menu, but we can add more buttons in the menu. To do so, we can enumerate several items in the `Menu::create` method and specify `nullptr` at the end. To add multiple buttons in the menu, use the following code:

[PRE1]

In addition, it is possible to add an item by using the `addChild` method of the menu instance.

[PRE2]

If the button is pressed, the lambda expression process that you specify when you create an instance of `MenuItemSprite` starts running. The argument is passed an instance of the `MenuItemSprite` that was pressed.

## There's more...

It is also possible to automatically align multiple buttons. We created three items in the `Resources/res` folder. These are named `item1.png`, `item2.png`, and `item3.png`. You can create three buttons and use the following code to align these buttons vertically in the center of the screen:

[PRE3]

![There's more...](img/0561_05_03.jpg)

If you want to align these items horizontally, you can use the following code:

[PRE4]

Until now, the alignment of intervals has been adjusted automatically; however, if you want to specify the padding, you can use another method.

The following code will specify the intervals side by side in a vertical manner:

[PRE5]

The following code will specify the intervals side by side in a horizontal manner:

[PRE6]

# Creating buttons

In this recipe, we will explain how to create buttons. Before the `Button` class was released, we created a button by using the `Menu` class that was introduced in the previous recipe. Due to the `Button` class, it has become possible to finely control the button press.

## Getting ready

To use the `Button` class and other GUI parts mentioned in this chapter, you must include the `CocosGUI.h` file. Let's add the following line of code in `HelloWorldScene.cpp`:

[PRE7]

## How to do it...

Let's create a button using the `Button` class. Firstly, you will generate a button instance by using `item1.png` image that was used in the previous recipe. We will also specify the callback function as a lambda expression by using the `addEventListener` method when the button is pressed. You can create the button by using the following code:

[PRE8]

## How it works...

You can now run this project and push the button. Further, you can move your touch position and release your finger. Thus, you will see that the touch status of the button will change in the log. Let's take a look at it step-by-step.

When you use the `Button` class and other GUI parts mentioned in this chapter, you have to include the `CocosGUI.h` file as this file defines the necessary classes. Further, please note that these classes have their own namespace such as "`cocos2d::ui`."

It is easy to create an instance of the `Button` class. You only need to specify the sprite file name. Further, you can create a callback function as a lambda expression by using the `addTouchEventListener` method. This function has two parameters. The first parameter is a button instance that was pressed. The second parameter is the touch status. Touch statuses are of four types. `TouchEventType::BEGAN` is the status at the moment that the button is pressed. `TouchEventType::MOVE` is the event type that occurs when you move your finger after you press it. `TouchEventType::ENDED` is the event that occurs at the moment you release your finger from the screen. `TouchEventType::CANCELED` is the event that occurs when you release your finger outside of the button.

## There's more...

It is possible to create a button instance by specifying the selected status image and the disabled status image. Create this button by using the code here.

[PRE9]

Unlike the `MenuItemSprite` class, you won't be able to specify the selection status by changing the normal image color that was set using the `setColor` method. You have to prepare the images as selected image and disabled image.

# Creating checkboxes

In this recipe, we will create a checkbox. In Cocos2d-x version 2, a checkbox was created by using the `MenuItemToggle` class. However, doing so was quite cumbersome. In Cocos2d-x version 3, we can create a checkbox by using the `Checkbox` class that can be used in Cocos Studio.

## Getting ready

So let's prepare the images of the checkbox before you start. Here, we have prepared the images of the required minimum `On` and `Off` status. Please add these images to the `Resouces/res` folder.

The `Off` status image will look something like this:

![Getting ready](img/0561_05_04.jpg)

The `On` status image will look something like this:

![Getting ready](img/0561_05_05.jpg)

## How to do it...

Let's create a checkbox by using the `Checkbox` class. First, you will generate a checkbox instance by using the `check_box_normal.png` image and the `check_box_active.png` image. You will also specify the callback function as a lambda expression by using the `addEventListener` method when the checkbox status is changed. Create the checkbox by using the following code:

[PRE10]

The following figure shows that the checkbox was selected by running the preceding code.

![How to do it...](img/0561_05_06.jpg)

## How it works...

It generates the instance of a checkbox by specifying the `On` and `Off` images. Further, the callback function was specified in the same way as the `Button` class was in the previous recipe. A checkbox has two `EventType` options, namely `ui::Checkbox::EventType::SELECTED` and `ui::Checkbox::EventType::UNSELECTED.`

You can also get the status of the checkbox by using the `isSelected` method.

[PRE11]

You can also change the status of the checkbox by using the `setSelected` method.

[PRE12]

## There's more...

In addition, it is possible to further specify the image of a more detailed checkbox status. The `Checkbox::create` method has five parameters. These parameters are as follows:

*   Unselected image
*   Unselected and pushing image
*   Selected image
*   Unselected and disabled image
*   Selected and disabled image

Here's how to specify the images of these five statuses:

[PRE13]

To disable the checkbox, use the following code:

[PRE14]

# Creating loading bars

When you are consuming a process or downloading something, you can indicate that it is not frozen by showing its progress to the user. To show such progresses, Cocos2d-x has a `LoadingBar` class. In this recipe, you will learn how to create and show the loading bars.

## Getting ready

Firstly, we have to prepare an image for the progress bar. This image is called `loadingbar.png`. You will add this image in the `Resouces/res` folder.

![Getting ready](img/0561_05_07.jpg)

## How to do it...

It generates an instance of the loading bar by specifying the image of the loading bar. Further, it is set to 0% by using the `setPercent` method. Finally, in order to advance the bar from 0% to 100% by 1% at 0.1 s, we will use the `schedule` method as follows:

[PRE15]

The following figure is an image of the loading bar at 100%.

![How to do it...](img/0561_05_08.jpg)

## How it works...

You have to specify one image as the loading bar image to create an instance of the `LoadingBar` class. You can set the percentage of the loading bar by using the `setPercent` method. Further, you can get its percentage by using the `getPercent` method.

## There's more...

By default, the loading bar will progress toward the right. You can change this direction by using the `setDirection` method.

[PRE16]

When you set the `ui::LoadingBar::Direction::RIGHT` value, the start position of the loading bar is the right edge. Then, the loading bar will progress in the left direction.

# Creating sliders

In this recipe, we will explain the slider. The slider will be used for tasks such as changing the volume of the sound or music. Cocos2d-x has a `Slider` class for it. If we use this class, we can create a slider easily.

## Getting ready

So, let's prepare the images of the slider before we start. Please add these images in the `Resouces/res` folder.

*   `sliderTrack.png`: Background of the slider![Getting ready](img/0561_05_09.jpg)
*   `sliderThumb.png`: Image to move the slider![Getting ready](img/0561_05_10.jpg)

## How to do it...

Let's create a slider by using the `Slider` class. First, you will generate a slider instance by using the `sliderTrack.png` image and the `sliderThumb.png` image. You will also specify the callback function as a lambda expression by using the `addEventListener` method when the slider value is changed.

[PRE17]

The following figure shows the result of the preceding code.

![How to do it...](img/0561_05_11.jpg)

## How it works...

You have to specify two images as the slider's bar image and the slider's thumb image to create an instance of the `Slider` class. The callback function was specified in the same way as the `Button` class was in the previous recipe. The slider has only one `EventType` as `ui::Slider::EventType::ON_PERCENTAGE_CHANGED`. That's why the status is the only changing value. You can get the percentage shown on the slider by using the `getPercent` method.

## There's more...

If you want to see the progress on the slider, you can use the `loadProgressBarTexture` method. We will require an image for the progress bar. The following image shows the progress bar image. Let's add it to the `Resources/res` folder.

![There's more...](img/0561_05_12.jpg)

Then, we use the `loadProgressbarTexture` method by specifying this image.

[PRE18]

Let's run the code that has been modified so far. You will see it with the color on the left side of the bar as shown in the following screenshot:

![There's more...](img/0561_05_13.jpg)

# Creating text fields

You may want to set a nickname in your game. To set nicknames or to get the player's input text, you can use the `TextField` class. In this recipe, we will learn about a simple `TextField` sample and how to add a textfield in a game.

## How to do it...

You will create a text field by specifying the placeholder text, font name, and font size. Then, you set a callback function by using `addEventListener`. In the callback function, you can get the text that the player input in the `textField`. Create the `textField` by using the following code:

[PRE19]

Let's run this code. You will see it within the placeholder text, and it will show the keyboard automatically as shown in the following screenshot:

![How to do it...](img/0561_05_14.jpg)

## How it works...

1.  You create an instance of the `TextField` class. The first argument is the placeholder string. The second argument is the font name. You can specify only a true type font. The third argument is the font size.
2.  You can get the event by using the `addEventListener` method. The following list provides the event names and their descriptions:

| Event Name | Description |
| --- | --- |
| `ATTACH_WITH_IME` | The keyboard will appear. |
| `DETACH_WITH_IME` | The keyboard will disappear. |
| `INSERT_TEXT` | The text was input. You can get the string by using the `getString` method. |
| `DELETE_BACKWARD` | The text was deleted. |

## There's more...

When a player enters a password, you have to hide it by using the `setPasswordEnable` method.

[PRE20]

Let's run the code that has been modified so far. You will see how to hide a password that you enter, as shown in the following screenshot:

![There's more...](img/0561_05_15.jpg)

# Creating scroll views

When a huge map is displayed in your game, a scroll view is required. It can be scrolled by a swipe, and bounce at the edge of the area. In this recipe, we explain the `ScrollView` class of Cocos2d-x.

## How to do it...

Let's implement it right away. In this case, we doubled the size of `HelloWorld.png`. Further, we try to display this huge image in `ScrollView`. Create the scroll view by using the following code:

[PRE21]

Let's run this code. You will see the huge `HelloWorld.png` image. Further, you will see that you can scroll it by swiping.

![How to do it...](img/0561_05_16.jpg)

## How it works...

1.  You create an instance of the `ScrollView` class by using the `create` method without arguments.
2.  You set the direction of the scroll view by using the `setDirection` method. In this case, we want to scroll up and down, and left and right, so you should set `ui::ScrollView::Direction::BOTH`. This implies that we can scroll in both the vertical and the horizontal directions. If you want to scroll just up and down, you set `ui::ScrollView::Direction::VERTICAL`. If you want to scroll just left and right, you set `ui::ScrollView::Direction::HORIZONTAL`.
3.  If you want to bounce when it is scrolled at the edge of the area, you should set `true` by using the `setBounceEnabled` method.
4.  You will provide the content to be displayed in the scroll view. Here, we have used `HelloWorld.png` that has been scaled twice.
5.  You have to specify the size of the content in the scroll view by using the `setInnerContainerSize` method. In this case, we specify double the size of `HelloWorld.png` in the `setInnerContainerSize` method
6.  You have to specify the size of the scroll view by using the `setContentSize` method. In this case, we specify the original size of `HelloWorld.png` by using the `setContentSize` method.

# Creating page views

A page view is similar to a scroll view, but it will be scrolled on a page-by-page basis. `PageView` is also a class in Cocos2d-x. In this recipe, we will explain how to use the `PageView` class.

## How to do it...

Let's immediately get it implemented. Here, we will arrange three images of `HelloWorld.png` side-by-side in the page view. Create the page view by using the following code:

[PRE22]

When you run this code, you will see one `HelloWorld.png`. You will see that you can move to the next page by using a swiping movement.

## How it works...

Create an instance of the `PageView` class by using the `create` method without arguments. Here, we set it as the same size as that of the screen.

Display three `HelloWorld.png` images side-by-side. You must use the `Layout` class to set the page layout in `PageView`.

Set the page size and add the image by using the `addChild` method.

Insert an instance of the `Layout` class to the page view by using the `insertPage` method. At this time, you specify the page number as the second argument.

Get the event when the page has changed, you use the `addEventListener` method. `PageView` has only one event, `PageView::EventType::TURNING`. You can get the current page number by using the `getCurPageIndex` method.

# Creating list views

`ListView` is a class in Cocos2d-x. It is like `UITableView` for iOS or `List View` for Android. `ListView` is useful for creating a lot of buttons as required in the case of setting a scene. In this recipe, we will explain how to use the `ListView` class.

## How to do it...

Here, we try to display `ListView` that has 20 buttons. Each button is identified with a number such as "`list item 10.`" In addition, we display the number of the button that you selected on the log when you tap any button. Create the list view by using the following code:

[PRE23]

When you run this code, you will see some buttons. You will see that you can scroll it by swiping and you can get the number of the button you tapped.

![How to do it...](img/0561_05_17.jpg)

## How it works...

1.  Create an instance of the `ListView` class. It is possible to specify the scroll direction in the same way as `ScrollView`. Since we want to scroll only in the vertical direction, you specify `ui::ListView::Direction::VERTICAL`. Also, you can specify the bounce at the edge of the area by using the `setBounceEnabled` method.
2.  Create 20 buttons to display in the list view. You have to use the `Layout` class to display the content in the list view as in the case of `PageView`. You add an instance of the `Button` class to the instance of the `Layout` class.
3.  Get the event by using the `addEventListener` method. `ListView` has two events, namely `ON_SELECTED_ITEM_START` and `ON_SELECTED_ITEM_END`. When you touch the list view, `ON_SELECTED_ITEM_START` is fired. When you release the finger without moving it, `ON_SELECTED_ITEM_END` is fired. If you move your finger, `ON_SELECTED_ITEM_END` is not fired and it will be a scrolling process. You can get the button number by using the `getCurSelectedIndex` method.