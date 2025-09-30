# Chapter 4. Building Scenes and Layers

The following topics will be covered in this chapter:

*   Creating scenes
*   Transitioning between scenes
*   Transitioning scenes with effects
*   Making original transitions for replacing scenes
*   Making original transitions for popping scenes
*   Creating layers
*   Creating modal layers

# Introduction

One screen has one scene. A scene is a container that holds Sprite, Labels, and other objects. For example, a scene can be a title scene, a game scene, or an option menu scene. Each scene has multiple layers. A layer is a transparent sheet similar to Photoshop's layer. Objects that added to layers are displayed on the screen. In this chapter, we will explain how to use the `Scene` class and the `Layer` class and how to transition between scenes. Finally, by the end of this chapter, you will be able to create original scenes and layers.

# Creating scenes

In Cocos2d-x, your games should have one or more scenes. A scene is basically a node. In this recipe, we will explain how to create and use a `Scene` class.

## How to do it...

In this recipe, we will use the project that was created in [Chapter 1](ch01.html "Chapter 1. Getting Started with Cocos2d-x"), *Getting Started with Cocos2d-x*.

1.  Firstly, duplicate the `HelloWorldScene.cpp` and `HelloWorldScene.h` files at `Finder` and rename them as `TitleScene.cpp` and `TitleScene.h`. Secondly, add them to the Xcode project. The result is shown in the following image:![How to do it...](img/B0561_04_01.jpg)
2.  Next, we have to change `HelloWorldScene` to `TitleScene` and place the search and replace method in the tips section.

    ### Tip

    **How to search for and replace a class name?**

    In this case, select `TitleScene.h` and then the **Find** | **Find and Replace …** menu in Xcode. Then, enter `HelloWorld` in the **String Matching** area and `TitleScene` in the **Replacement String** area. Execute all replacements. Follow the same process for `TitleScene.cpp`. The result is the following code for `TitleScene.h`:

    The result obtained for `TitleScene.h` is as follows:

    [PRE0]

    Next, the result for `TitleScene.cpp` is as follows:

    [PRE1]

3.  Next, add a label to the difference between `TitleScene` and `HelloWorldScene`. Add it before the return line in the `TitleScene::init` method as follows:

    [PRE2]

4.  Similarly, add the label in the `HelloWorld::init` method.

    [PRE3]

5.  Next, to display the `TitleScene` class, change `AppDelegate.cpp` as follows:

    [PRE4]

The result is shown in the following image:

![How to do it...](img/B0561_04_02.jpg)

## How it works...

First, you need to create `TitleScene` by duplicating the `HelloWorldScene` class files. It is pretty difficult to create an original `Scene` class from a blank file. However, a basic class of `Scene` is patterned. So, you can create it easily by duplicating and modifying the `HelloWorldScene` class files. While you are developing your game, you need to execute this step when you need a new scene.

Finally, we change the `AppDelegate.cpp` file. The `AppDelegate` class is the first class to be executed in Cocos2d-x. The `AppDelegate:: applicationDidFinishLaunching` method is executed when the application is ready to run. This method will prepare the execution of Cocos2d-x. Then, it will create the first scene and run it.

[PRE5]

The `TitleScene::createScene` method is used to create a title scene, and the `runWithScene` method is used to run it.

# Transitioning between scenes

Your games have to transition between scenes. For example, after launching your games, the title scene is displayed. Then, it is transitioned into the level selection scene, game scene, and so on. In this recipe, we will explain how to facilitate transition between scenes, which would improve the gameplay and the flow of the game.

## How to do it...

A game has a lot of scenes. So, you might need to move between scenes in your game. Perhaps, when a game is started, a title scene will be displayed. Then, a game scene will appear in the next title scene. There are two ways to transition to a scene.

1.  One is to use the `Director::replaceScene` method. This method replaces a scene outright.

    [PRE6]

2.  The other is to use the `Director::pushScene` method. This method suspends the execution of the running scene and pushes a new scene on the stack of the suspended scene.

    [PRE7]

In this case, the old scene is suspended. You can get back to the old scene to pop up a new scene.

[PRE8]

## How it works...

Layer, Sprite, and other nodes can be displayed by using the `addChild` method. However, Scene cannot be displayed by using the `addChild` method; it can be displayed by using the `Director::replaceScene` or `Director::pushScene` methods. That's why `Scene` is visible only on one screen at the same time. `Scene` and `Layer` are similar, but there is a significant difference.

Usually, you will use the `replaceScene` method when you change the scene from the title scene to the game scene. Further, you can use the `pushScene` method to display a modal scene, as in the case of pausing a scene during a game. In this case, an easy way to suspend a game scene is to pause the game.

### Tip

When scenes are replaced in a game, the applications will release the memory used by the old scenes. However, if games push scenes, they will not release the memory used by the old scenes because it will suspend it. Further, games are resumed when new scenes are popped. If you added a lot of scenes by using the `pushScene` method, the device memory will no longer be enough.

# Transitioning scenes with effects

Popular games display some effects when transitioning scenes. These effects can be natural, dramatic, and so on. Cocos2d-x has a lot of transitioning effects. In this recipe, we will explain how to use a transitioning effect and the effect produced.

## How to do it...

You can add visual effects to a scene transition by using the `Transition` class. Cocos2d-x has many kinds of `Transition` classes. However, there is only one pattern for how to use them.

[PRE9]

It can be used when a scene was pushed.

[PRE10]

## How it works...

Firstly, you need to create the `nextscene` object. Then, you need to create a `transition` object with a set duration and an incoming scene object. Lastly, you need to run `Director::pushScene` with the `transition` object. This recipe sets the duration for the transition scene and the fade action as one second. The following table lists some of the major `Transition` classes:

| Transition Class | Description |
| --- | --- |
| `TransitionRotoZoom` | Rotates and zooms out of the outgoing scene, and then, rotates and zooms into the incoming scene. |
| `TransitionJumpZoom` | Zooms out and jumps the outgoing scene, and then jumps and zooms into the incoming scene. |
| `TransitionMoveInL` | Moves scene in from right to the left. |
| `TransitionSlideInL` | Slides in the incoming scene from the left border. |
| `TransitionShrinkGrow` | Shrinks the outgoing scene while enlarging the incoming scene. |
| `TransitionFlipX` | Flips the screen horizontally. |
| `TransitionZoomFlipX` | Flips the screen horizontally by doing a zoom out/in. The front face shows the outgoing scene, and the back face shows the incoming scene. |
| `TransitionFlipAngular` | Flips the screen half horizontally and half vertically. |
| `TransitionZoomFlipAngular` | Flips the screen half horizontally and half vertically by zooming out/in a little. |
| `TransitionFade` | Fades out of the outgoing scene, and then, fades into the incoming scene. |
| `TransitionCrossFade` | Cross-fades two scenes by using the `RenderTexture` object. |
| `TransitionTurnOffTiles` | Turns off the tiles of the outgoing scene in an random order. |
| `TransitionSplitCols` | The odd columns go upwards, while the even columns go downwards. |
| `TransitionSplitRows` | The odd rows go to the left, while the even rows go to the right. |
| `TransitionFadeTR` | Fades the tiles of the outgoing scene from the left-bottom corner to the top-right corner. |
| `TransitionFadeUp` | Fades the tiles of the outgoing scene from the bottom to the top. |
| `TransitionPageTurn` | Peels back the bottom right-hand corner of a scene to transition to the scene beneath it, thereby simulating a page turn. |
| `TransitionProgressRadialCW` | Counterclockwise radial transition to the next scene. |

## There's more...

You can also learn the beginning and the end of the transition scene by using the `onEnterTransitionDidFinish` method and the `onExitTransitionDidStart` method. When your game shows the new scene completely, the `onEnterTransitionDidFinish` method is called. When the old scene starts disappearing, the `onExitTransitionDidStart` method is called. If you'd like to do something during the time that the scenes appear or disappear, you will need to use these methods.

Let's now look at an example of using the `onEnterTransitionDidFinish` and `onExitTransitionDidStart` methods. `HelloWorldScene.h` has the following code:

[PRE11]

# Making original transitions for replacing scenes

You know that Cocos2d-x has a lot of transitioning effects. However, if it does not have an effect that you need, it is difficult to create an original transitioning effect. However, you can create it if you have the basic knowledge of transitioning effects. In this recipe, we will show you how to create original transitions.

## How to do it...

Even though Cocos2d-x has a lot of different types of `Transition` classes, you may not find a transition effect that you need. In this recipe, you can create an original transition effect such as opening a door. When the replacement of a scene begins, the previous scene is divided into two and open to the left or right.

You have to create new files named "`TransactionDoor.h`" and "`TransactionDoor.cpp`," and add them to your project.

`TransactionDoor.h` has the following code:

[PRE12]

Use the following code for `TransactionDoor.cpp`:

[PRE13]

The following code will allow us to use the `TransitionDoor` effect:

[PRE14]

## How it works...

All types of transitions have `TransitionScene` as `SuperClass`. `TransitionScene` is a basic class and has a basic transition process. If you would like to create the original transition effect in an easier way, you would look for a similar transition effect in Cocos2d-x. You can then create your class from a similar class. The `TransitionDoor` class is created from the `TransitionSplitCol` class. Then, add and modify them where necessary. However, it is necessary to have basic knowledge about them in order to fix them.

The following are some of the important properties of the `Transition` class:

| Properties | Description |
| --- | --- |
| `_inScene` | Pointer of the next scene. |
| `_outScene` | Pointer of the out scene. |
| `_duration` | Duration of the transition, a float value specified by the create method. |
| `_isInSceneOnTop` | Boolean value; if it is true, the next scene is the top of the scene graph. |

Some of the important properties of the `transition` class are as follows:

| Properties | Description |
| --- | --- |
| `onEnter` | To start the transition effect. |
| `Action` | To create an effect action. |
| `onExit` | To finish the transition effect and clean up. |

In the case of the `TransitionDoor` class, the next scene is set to be visible and the previous scene in split into two grids in the `onEnter` method. Then, an effect such as opening a door is started. In the action method, an instance of the `Action` class is created by using the `SplitDoor` class. The `SplitDoor` class is based on the `SplitCol` class in Cocos2d-x. The `SplitDoor` class moves two grids of the previous scene to the left or the right.

## There's more...

There are some necessary methods in addition to those described above. These methods are defined in the `Node` class.

| Properties | Description |
| --- | --- |
| `onEnter` | Node starts appearing on the screen |
| `onExit` | Node disappears from the screen |
| `onEnterTransitionDidFinish` | Node finishes the transition effect after appearing on the screen |
| `onExitTransitionDidStart` | Node starts the transition effect before disappearing from the screen |

If you want to play background music when the scene appears on the screen, you can play it by using the `onEnter` method. If you want to play it before finishing the transition effect, use the `onEnterTransitionDidFinish` method. Other than these, the initial process in the `onEnter` method starts the animation in the `onEnterTransitionDidFinish` method, cleans up the process in the `onExit` method, and so on.

# Making original transitions for popping scenes

Cocos2d-x has transition effects for pushing a scene. For some reason, it does not have transition effects for popping scenes. We'd like to transition with an effect for popping scenes after pushing scenes with effects. In this recipe, we will explain how to create an original transition for popping scenes.

## Getting ready

In this recipe, you will understand how to pop a transition scene with effects. You will need a new class, so you have to make new class files called `DirectorEx.h` and `DirectorEx.cpp` and add them to your project.

## How to do it...

Cocos2d-x has a transition scene with effects for pushing scenes. However, it does not have transition effects for popping scenes. Therefore, we create an original class called `DirectorEx` to create a transition effect for popping scenes. The code snippet is given next.

`DirectorEx.h` has the following code:

[PRE15]

`DirectorEx.cpp` has the following code:

[PRE16]

This class can be used as follows:

[PRE17]

## How it works...

If we customized the `Director` class in Cocos2d-x, it can transition with the effect for popping a scene. However, this is not a good idea. Therefore, we create a sub-class of the `Director` class called the `DirectorEx` class and use this class as follows:

1.  Firstly, you can get an instance of the `DirectorEx` class to cast an instance of the `Director` class.

    [PRE18]

2.  Further, you have to get an instance of the previous scene.

    [PRE19]

3.  Next, you have to create a transition effect.

    [PRE20]

4.  Finally, you can pop a scene with this effect by using the `DirectorEx::popScene` method.

    [PRE21]

# Creating layers

A layer is an object that can be used on `Scene`. It is a transparent sheet similar to Photoshop's layer. All the objects are added to `Layer` in order to be displayed on the screen. Further, a scene can have multiple layers. Layers are also responsible for accepting inputs, drawing, and touching. For example, in the game, a scene has a background layer, hud layer, and a player's layer. In this recipe, we will explain how to use `Layer`.

## How to do it...

The following code shows how to create a layer and add it to a scene:

[PRE22]

That's easy. If you have a color layer, you can do it.

[PRE23]

## How it works...

The Scene class is the one displayed on the screen, but the `Layer` class can be stacked in many layers. Scene has one or more layers, and Sprite has to be on a layer. The `Layer` class is a transparent sheet. In addition, a transparent node needs more CPU power. So, you need to be careful not to stack too many layers.

# Creating modal layers

In user interface design, a modal layer is an important layer. A modal layer is like a child window. When a modal layer is showing, players cannot touch any other button outside the modal layer. They can only touch the button on the modal layer. We will need modal layers when we confirm something with the players. In this recipe, we will explain how to create modal layers.

## How to do it...

Firstly, you have to two new files named `ModalLayer.h` and `ModalLayer.cpp`. They should contain the following code:

`ModalLayer.h` should have the following code:

[PRE24]

You should create a sub-class from the `ModalLayer` class and add a menu button or some design that you need. You then have to create an instance of it and add it to the running scene. Then, it should enable the buttons on the modal layer but disable the buttons at the bottom of the modal layer.

[PRE25]

## How it works...

It is easy to create a modal layer in Cocos2d-x version 3\. In version 3, a touch event occurs from the top of the layer. So, if the modal layer picks up all the touch events, the nodes under the modal layer are notified of these. The modal layer is picking up all of the events. Refer to the following code:

[PRE26]

### Tip

This modal layer can pick up all touching events. However, Android has key events like the back key. When a player touches the back key when the modal layer is displayed, you have to decide to do it. In one of the cases, the modal is closed, and in another, the back key is ignored.