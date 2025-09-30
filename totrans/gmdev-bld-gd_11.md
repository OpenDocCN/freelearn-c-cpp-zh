# 11

# Creating the User Interface

To start this chapter, let’s begin by asking a simple question: what was the first multiplayer game you played?

If you are thinking of a PC or a console game, try thinking another way. Imagine a bunch of kids holding their arms out, pretending to shoot and take down the bad guys invading their neighborhood. Perhaps there was an evocative action movie the night before on TV. Now, these kids are bringing to life what they think is possible within the realm of physics, mixed with a bit of fantasy and what they remember from the movie. Some kids will even pretend they have been harmed along the way. Fallen comrades will be avenged in the end, and good will once again prevail against evil. Who’s keeping the score here – that is, who has how many hit points?

How about the servers, internet speed, and likewise? Did the kids even need a **user interface** (**UI**) to play their game? No, because it was still easy for them to keep track of what was happening. But when the number of things people need to pay attention to gets beyond a certain point, it gets overwhelming. In other words, a UI is needed when using a system without one becomes impractical.

This is not unique to video games. In the real world, you use an ATM to access your bank accounts. The information and functions you need will be presented in a clear, concise manner; checking your accounts, sending e-transfers, and accessing the current interest rates are quick and easy to do all from one place, thanks to a well-designed UI.

In our game, despite what Clara expected, her uncle was not there but had left a note on the pier. We need a way for the player to access this information. Thus, in this chapter, we’ll present a few of the UI components Godot has in its arsenal to convey this message. We’ll start with a simple **Button** node, followed by a **Panel** component. In this panel, we will display some text via the **Label** component.

While you are adding more and more UI elements to the game, you’ll also learn how to apply styles to these so that they look more like they belong to the game world. After all, the default ones have that default gray look, which might be better suited for prototyping.

Styling Godot nodes may feel tiresome after you do it more than a few times, especially if you are doing it for the same kind of button with different text. As a solution, we’ll demonstrate how to take advantage of themes, which is a powerful tool that will help you in your styling efforts.

As usual, we’ll be discussing a few relevant side topics that are pertinent to the creation of UIs. With that in mind, in this chapter, we will cover the following topics:

*   Creating a simple button
*   Wrapping in a panel
*   Filling the panel with more control nodes
*   Taking advantage of themes

By the end of this chapter, you’ll have learned how to exploit UI nodes to help the player read the note that Clara’s uncle had left for her.

# Technical requirements

If you think you don’t have enough artistic talent to create UIs, then rest assured for two reasons. First, we’ll mainly focus on utilizing the UI components in Godot. Therefore, the graphic design aspect won’t be our concern. Second, we are providing you with the necessary assets in the `Resources` folder in `Chapter 11` of this book’s GitHub repository. Inside it, you’ll find two folders: `Fonts` and `UI`. Simply merge these two folders into your Godot project folder.

This book’s GitHub repository, [https://github.com/PacktPublishing/Game-Development-with-Blender-and-Godot](https://github.com/PacktPublishing/Game-Development-with-Blender-and-Godot), contains all the assets you need. Lastly, you can either continue your work from the previous chapter or utilize the `Finish` folder from `Chapter 10`.

# Creating a simple button

A UI is a collection of components you lay out in a coherent manner around the core visuals of your game. The most essential UI component to start with may have been a **Label** node if we wanted it to be similar to printing “Hello, world!” when we are learning a new programming language. However, we’ll start with a **Button** node since the former case is so trivial, and we can also learn how to style a **Button** during this effort.

Before we start throwing around a bunch of UI nodes willy-nilly, we should first mention the right kind of structure to hold our UI nodes. We can use **CanvasLayer** similar to using a **Spatial** node to nest other nodes such as **MeshInstance**, **AnimationPlayer**, and others.

We’ve already been creating scenes mainly to display 3D models. Let’s follow similar steps for the sake of creating the UI:

1.  Create a blank scene and save it as `UI.tscn` in the `Scenes` folder.
2.  Choose `UI`.
3.  Attach a `Close`.
4.  Type `Close` for its **Text** value in the **Inspector** panel.

There’s nothing fancy going on so far, but we now have a button aligned, by default, to the top left of the viewport. The width of this button also expanded when you were typing the text it displays.

Control versus CanvasLayer

We mentioned that a **Spatial** node would be the root node for 3D nodes. So, for the sake of keeping things familiar, we could have used a **Control** node to hold the **Button** node. Rest assured, you could still inject a **Control** node inside a **CanvasLayer**. The real reason we used a **CanvasLayer** as the root is for its **Layer** property in the **Inspector** panel. By changing the value of this, you can change the draw order, which means you can decide which **CanvasLayer** will render first. This is a useful mechanism when you have multiple UI structures that need to be layered on top of each other in a precise order.

The button we have just added looks boring. It doesn’t quite fit the world we are creating. Now, let’s use a custom graphic asset to style our button:

1.  Expand the **Styles** subsection in the **Theme Overrides** section of the **Inspector** panel.
2.  Using the dropdown for the **Normal** property, select the **New StyleBoxTexture** option.
3.  Click the **StyleBoxTexture** title as it will populate the **Inspector** panel with its properties.
4.  Drag `button_normal.png` from `UI` into the **FileSystem** panel and drop it in the **Texture** property.
5.  Expand the `8` for all the margin values.
6.  Press *F6* to launch the `UI.tscn` scene and try to interact with the button.

You have taken quite a few steps to style a simple button, so let’s break down what’s happened.

In *step 1*, you told Godot that you wanted to override the default theme, which was giving that gray look to the button. Without user interaction, the button will be in its normal state; so, that’s what you intend to change in *step 2*. We’ll discover how to change the other states very soon.

*Step 3* was about defining the properties of this `Lorem ipsum dolor sit amet`. Notice how the button is getting wider without looking stretched and keeping the rounded corners intact. This needs a proper explanation.

Setting margins involves doing more than just accommodating text. Carefully selected values will make sure the texture will enlarge or shrink as needed without losing some of its qualities, such as rounded corners. When the asset has rounded corners, if the texture is stretched, you will end up with a distorted look. The practice of conserving the core features of a texture and allowing it to be resized properly without distortion is called 9-slice scaling. You can learn more about it here: [https://en.wikipedia.org/wiki/9-slice_scaling](https://en.wikipedia.org/wiki/9-slice_scaling).

When you launched the `UI.tscn` scene in *step 6*, the button must have shown its normal state as a brown texture. If you move your mouse over it, you’ll see that the button will show the default look again because you haven’t set the hover state yet. This can be seen in the following screenshot:

![Figure 11.1 – The button only has its normal state styled ](img/Figure_11.1_B17473.jpg)

Figure 11.1 – The button only has its normal state styled

Similar to the way you assigned a texture to the normal state of the button, you can do so for the other states. Let’s do this for the hover state:

1.  Select the **Close** button in the **Scene** tree.
2.  Assign a **New StyleBoxTexture** to the **Hover** state in the **Styles** subsection under **Theme Overrides** and click this **StyleBoxTexture** to set its properties.
3.  Drag `button_hover.png` from the `UI` folder and set the margins to `8`.
4.  Press *F6* and move your mouse over the button.

We’ll repeat this effort for the pressed and disabled states as well. We won’t use disabled buttons in our game, but why not be thorough? Also, in most scenarios, you can repurpose the pressed state for the focus state. The different results are shown in the following screenshot:

![Figure 11.2 – The normal, hover, pressed, and disabled states of a button with a custom texture ](img/Figure_11.2_B17473.jpg)

Figure 11.2 – The normal, hover, pressed, and disabled states of a button with a custom texture

Before we move on to introducing more UI nodes, we suggest that you change the text of that button back to `Close` since we’ll use this button to close a panel that will simulate a note from Clara’s uncle. Speaking of which, it’s time to learn what was written in that note.

# Wrapping in a panel

So far, we have created a button and styled it. However, it would be nice if it served some purpose, especially since we gave it a meaningful label. We’ll write some code so that this button can close a panel near the end of the *Filling the panel with more control nodes* section. Before we get to that point, though, we need the panel.

As we are introducing more UI nodes, let’s remember why we are doing this within the game’s context. Clara’s uncle had left a note. We’ll simulate that note with a combination of UI nodes in Godot so that it looks as follows:

![Figure 11.3 – Clara’s note ](img/Figure_11.3_B17473.jpg)

Figure 11.3 – Clara’s note

We’ve already taken care of the button, but it is currently sitting in the middle of nowhere. We’ll wrap it in a **Panel** node in this section after we give a short disclaimer.

A **Panel** node is just another **Control** node in Godot that usually holds other components. There is a similarly named node, **PanelContainer**, which might be confusing for beginners. The **Panel** node derives from the **Control** class, whereas the **PanelContainer** node inherits from the **Container** class. Also, the **Container** class inherits from the **Control** class. This kind of technical detail might be important when you are doing more advanced work. We won’t, so either one would work fine for our intents and purposes in this book. Therefore, we’ll stick with the **Panel** node.

At this point, we are ready to add a **Panel** node and style it:

1.  Add a **Panel** node under the root **UI** node in the **Scene** tree.
2.  Expand the **Rect** section in the **Inspector** panel.
3.  For the `600` for **X**.
4.  Type `400` for **Y**.

*   Assign a **New StyleBoxTexture** to the **Panel** property in the **Styles** subsection under **Theme Overrides**.*   Drag the **Close** button over the **Panel** node in the **Scene** panel so that the **Close** button is nested.

At this point, you should have the following output:

![Figure 11.4 – The paper texture has been simulated with the help of a Panel node ](img/Figure_11.4_B17473.jpg)

Figure 11.4 – The paper texture has been simulated with the help of a Panel node

We are getting closer and closer to the desired design we imagined for the note. The button in the panel is still aligned to the top left. You can drag it to a position that makes sense, but it might be easier to decide on that if you have some text within the panel. That’s what we’ll take care of next.

# Filling the panel with more control nodes

The uncle’s note is slowly taking shape. We’ll introduce a **Label** node in this section for the text portion. Additionally, we’ll have to figure out how to position all these elements so that the note resembles the layout we’d like to have. Lastly, we’ll discuss a few complementary **Control** nodes you may want to use in some other scenarios.

After all, we will still employ the most basic UI node: **Label**. If we had used it at the beginning, it would have looked unimpressive with its default style and color. Since we now have a proper texture over which this **Label** node can go, things will look more interesting. Follow these steps to do this:

1.  Select the **Panel** node in the **Scene** panel.
2.  Add a **Label** node and turn its **Autowrap** property on in the **Inspector** panel.
3.  Set its **Text** to the following:

`My dear Clara,`

`A close friend of mine is in dire need of help. I must leave immediately.`

`Check out the backpack by the decrepit cart. Inside, you will find a key to upstairs. Make yourself at home.`

`Your uncle, Bert`

Our last effort will result in an awkwardly tall text block. To remedy this, we could manually give some width and height to the **Label** node we have just inserted. While we are doing that, we could also change its position to make it look centered and have some margins off each edge. However, we can do something smarter: we can wrap this **Label** inside a **MarginContainer** that will set margins and automatically resize the text for us.

## Adding a MarginContainer

At this point, adding new nodes to the **Scene** panel must be a common task for you. Nevertheless, there are times, such as now, when deciding where to add a new node and what to nest in it might not be obvious. The question is, where can we add **MarginContainer**? Outside the **Panel** node or inside?

A **MarginContainer** is a specialized container that’s responsible for introducing margins so that its children look like they have padding. If we wrap the **Panel** node inside a **MarginContainer**, since the **Panel** node is holding the text, the whole structure, including the button, will be padded. That’s not good since we would like the text to have some space between its edges and the borders of the texture that constitutes the **Panel** node. Thus, this is what you need to do to only pad the text:

1.  Add a **MarginContainer** node inside the **Panel** node and nest **Label** inside this **MarginContainer** node.
2.  Set the following values in the `0` and both `1`.
3.  In the `0`.
4.  In the `60`.

We touched on a lot of terms in the preceding operation. The first two sets of actions, where we alter the values of anchor and margin, are not specific to a **MarginContainer**. They exist for every type of **Control** node. You can also see this fact as these properties were listed under the **Control** header in the **Inspector** panel.

The anchor and margin values we chose are such special values that we could have used a shortcut to achieve the same result. It would be selecting the **Full Rect** option in the expanded menu after you click the **Layout** button in the header section of the 3D viewport. This **Layout** button is visible in the following screenshot, just above the top-right corner of the paper texture.

We’ll use another option under that menu when we adjust the location of the **Close** button later. For now, compare your work to what you can see in the following screenshot:

![Figure 11.5 – The text now has padding, although it’s hard to read ](img/Figure_11.5_B17473.jpg)

Figure 11.5 – The text now has padding, although it’s hard to read

What was essential in the properties of that **MarginContainer** was adjusting its content margin values in the **Constants** subsection. That gave the text some room and positioned it correctly over the paper texture.

It’s a bit difficult to read the text, though. So, let’s see how we can make it legible and, even better, make it look like *Figure 11.3*.

## Styling the Label node

Although **MarginContainer** is now occupying as much space as the **Panel** node, and it’s providing margins to the text it’s holding, the text itself is hardly legible since it’s small and white over a lightly colored surface. Also, the font choice is wrong because it’s using the default font provided by Godot Engine. We’ll learn how we can fix all these issues in this section.

Let’s start by selecting the **Label** node in the **Scene** panel so that we can make some changes under **Theme Overrides**:

1.  Turn on the **Font Color** option in the **Colors** subsection. The color can be left black.
2.  Choose the `Kefario.otf` from **FileSystem** to the **Font Data** property in the **Font** subsection.
3.  Change `28` in the **Settings** subsection.

We’ll discuss what’s happened shortly, but here is what we have done so far:

![Figure 11.6 – The Label node now looks more like handwritten text ](img/Figure_11.6_B17473.jpg)

Figure 11.6 – The Label node now looks more like handwritten text

The default black color for the text seems to be fine, but you could pick a different color if you wish. A much more drastic change happened when we introduced a font type. We did this in two steps. First, we picked a **DynamicFont** type, which is slower than the other option, **BitmapFont**, but it lets you change the properties of the font at runtime. However, this is not enough to render a font since it works like a wrapper. So, you need to assign the font you would like to render. That’s what we did when we assigned a font file to the **FontData** property.

There is an important caveat we think you should be aware of with fonts since they are made of individual elements called glyphs. You can think of them as the letters in an alphabet. Not every font supports the full spectrum of an alphabet. For example, in the note UI that we designed, if you replace the text `you will` with its shortened form, `you’ll`, the apostrophe won’t render because it doesn’t exist as a glyph in the font. Usually, paid fonts come with a bigger set of glyphs. Otherwise, keep searching for free options with a more complete set.

Pixels versus points

When we chose 28 as the font size, that number was measured in pixels. In some graphics or text editors, you’ll often find fonts measured in points. This is something you have to be cautious about because if you transfer the numbers verbatim to Godot, your font will be rendered quite differently. So, mind your units.

In the real world, a note from Clara’s uncle would only contain the text portion. Thus, it would be absurd to expect a close button on top of an actual piece of paper. However, this is a game, and we’ve already discussed how UIs mix reality with functionality. To complete the UI for the note, it’s time we positioned that button.

## Positioning the Close button

We used a nice trick to position the text concerning the piece of paper it’s on. Can we replicate this for the **Close** button? Since a button can’t be considered a wide structure, we can’t stick it inside **MarginContainer**. However, we can still position it relative to the **Panel** node.

In the *Adding a MarginContainer* section, we used a longer method to adjust the dimensions of that component. We also mentioned that we would use a shortcut. This is how you can use it after selecting the **Close** button in the **Scene** panel:

1.  Expand the **Layout** menu and select the **Bottom Right** option.
2.  Hold down *Shift* and press the left and up arrow keys on your keyboard four times for each.

This will position the **Close** button at the bottom right corner, then pull it up and move it left just enough that it stays there. We mean it when we claim that it’ll be staying there. For example, select the **Panel** node, then try to resize it using the handles in the viewport. Does the button stay nicely tucked in that bottom right corner? Good! How about the **Label** node? Does the text flow to occupy the extra space? Neat!

Our efforts to develop what you saw in *Figure 11.3* are coming to fruition, as shown here:

![Figure 11.7 – Everything in the UI is positioned carefully ](img/Figure_11.7_B17473.jpg)

Figure 11.7 – Everything in the UI is positioned carefully

If you want to test your scene, go ahead and press *F6*. Depending on your setup, you may notice that the **Close** button will not be functional since it’s behind **MarginContainer**. So, try to resort the nodes in the **Scene** panel by dragging the nodes up and down. When you have the **Close** button after **MarginContainer**, everything should be functional.

Speaking of functionality, we haven’t wired anything up for the **Close** button. Ideally, that button should turn the visibility of the **Panel** off so that the note looks as if it’s been closed. Let’s do that next.

## Adding the close functionality

There are multiple ways we can attack this problem. We are going to show you one for brevity’s sake so that you can see what’s involved. You may have to apply similar principles differently in your future projects.

For example, the way we are treating the `UI.tscn` scene so far is to have one big **Panel** node as a direct child. Your games may need a lot more UIs with elements permanently visible on the screen, more notes to open and close, inventory screens with expanding parts to reveal more details, and likewise. There are many possibilities, which is why there are different types of architectures you can construct. There will always be a tradeoff between these different options, so we suggest you experiment with the benefits of different UI structures if you have some spare time.

Without further ado, our suggestion for implementing the closing functionality is to add a small script to the **Close** button. Select it and do the following:

1.  Attach a script to the `ButtonClose.gd` in the `Scripts` folder.
2.  Make this script file look as follows:

    ```cpp
    extends Button
    func _ready():
        connect("pressed", self, "on_pressed")

    func on_pressed():
        get_parent().visible = false
    ```

This architecture assumes that the button will always be the direct child of a node, so once it’s pressed, it will make its parent invisible. Ouch!

The benefit of this kind of simple structure is the convenience that the button doesn’t need to know the node structure it’s in. There is also a more conventional way of attaching the *pressed* behavior by using the **Node** panel and binding a signal. Either way is fine.

Constructing and improving UI elements may easily turn into a project by itself. You might be tempted to create that perfect setup for all future possible scenarios but keep in mind that overoptimization is a thing. Later, you may realize that you didn’t need all that preparation in the first place. We’ll talk about a similar situation next, where the note might be longer.

## Wrapping up

We now have a fully functional UI for displaying the note from Clara’s uncle, Bert. What if Bert had more to say? For example, let’s say the message had an extra line after his name, as shown here:

`Your uncle, Bert`

`P.S. I think I might have left my pet snake unattended. It might be wandering around, so be careful!`

If you were to add this extra text to the end of the **Label** node, the text would get uncomfortably close to the top and bottom of the paper texture. Similarly, imagine that this text block needed to be even longer, which is the case in some types of games where exposition is important. For instance, it is very common when displaying the details of a quest or an item in role-playing games.

Currently, we can make do by adjusting the font size of the text or making the margins narrower to allow more room for the new text. However, in more extreme situations, it might be better to use a **ScrollContainer** node. Just like you wrapped the **Label** node inside **MarginContainer**, you can wrap a **ScrollContainer** around the **Label** node, and tweak a few things to have a scrollable text block.

Coming up with the correct level of *nestedness* and deciding on the type and order of UI components is sometimes an effort of trial and error. Consequently, there aren’t any set formulas. Therefore, you may find yourself practicing and seeing what works best in your use case.

That being said, generalizing your efforts to maintain a consistent look and feel across your many UI nodes might be helpful. We’ll tackle themes next to accomplish this.

# Taking advantage of themes

Using or, more specifically, creating themes in your projects is smart on many accounts. First, we’ll discuss their usefulness, show you a few visual examples, and then create one for practicing. Let’s start with the reasons why you should use themes.

Firstly, using themes will save you from manually applying overrides to the components the way you’ve done so far. It’s still possible to keep adding manual touches here and there, but what would happen if you wanted to change a button’s artistic direction completely? This would trigger a chain reaction to change the look of other components too. So, you’d have to restart the manual editing. Furthermore, the ultimate worst-case scenario would be to revert your changes because, you know, we are human and we kind of tend to stick with our first choices more often than not.

Secondly, you could have multiple themes at the ready in your game. Although a button is still just a button, you could assign it one theme out of many. This will make that button look like it belongs to a family of components. Thus, your UI elements will have a consistent style.

Lastly, changing themes at runtime is a possibility. This means that if, in your game or the application you are building with Godot, you would like to swap themes for special occasions such as Christmas, this is entirely possible. Also, more and more desktop applications are being built with Godot. These could also benefit from theme swapping to offer their user the best choice. Godot Engine itself allows you to change themes. You can access the existing themes by opening **Editor Settings** and trying out a few themes. For example, try out the **Solarized (Light)** theme. Are you getting Unity vibes?

Changing a theme is not always about changing the colors of buttons or font sizes. For example, [https://365psd.com/psd/ui-kit-54589](https://365psd.com/psd/ui-kit-54589) and [https://365psd.com/day/3-180](https://365psd.com/day/3-180) are two UI kits we picked to show how drastically different your Godot UI nodes could look. *Figure 11.8* presents these two UI kits side by side:

![Figure 11.8 – Two distinct UI kits that are good candidates for themes ](img/Figure_11.8_B17473.jpg)

Figure 11.8 – Two distinct UI kits that are good candidates for themes

Since we have already seen how to change the look and feel of three types of nodes, **Button**, **Panel**, and **Label**, we’ll focus on other types of **Control** nodes. We’ll accomplish this in the context of creating a new theme.

## Creating a new theme

Since game development is an iterative process, planning every single thing ahead of time may not always be possible, and even be fruitless. That’s why it’s typical if you start by changing the UI nodes manually. Still, starting with a new theme and changing the properties of this theme may also be a good idea. Why? Because if your experiments for individually modifying the components yield a successful result, you won’t have to repeat what you have done in the theme. By creating a theme at the beginning, you’re building up as you go.

Also, creating a theme is like creating any other type of resource in Godot. We can do this by following a few simple steps:

1.  Right-click the `UI` folder in the `Themes` as its name.
2.  Right-click `Themes` in **FileSystem** and select the **New Resource** option.
3.  Choose `Dark.tres`.

This will create a **Theme** resource in your project. It should also enable a new panel in the bottom area that will show the preview of this new theme. As you make changes to your theme, updates can be previewed in this area since it might be faster to monitor your progress this way rather than adding and removing test components to/from your scene.

If the preview area looks small, it’s possible to enlarge it by clicking an icon next to Godot’s version number. This icon will look like two upward-facing arrows with a horizontal line above them. Press that and the theme preview will occupy the viewport. In the end, your editor will look as follows:

![Figure 11.9 – The theme preview has been expanded ](img/Figure_11.9_B17473.jpg)

Figure 11.9 – The theme preview has been expanded

By the way, the preview area is not static. You can interact with those UI components. It’s like a Godot scene running inside Godot. Now, we will modify the theme for the **CheckButton**, **CheckBox**, and **VSlider** components. We’ll also show a special state of the **CheckBox** node, also known as a radio button, in web development. However, our first candidate is **CheckButton**.

## Styling a CheckButton

The graphics assets we’ll be using to construct the new theme is the *Dark UI Kit*, which you can find at [https://365psd.com/psd/dark-ui-kit-psd-54778](https://365psd.com/psd/dark-ui-kit-psd-54778). We’ve already exported the necessary parts into the `UI` folder for you.

The theme we created is still the default theme, so it still shows the default components. We’ll have to pick the one we would like to change. This is how we do it:

1.  Press the button with the plus (**+**) icon in it. This is in between the **Manage Items** and **Override All** buttons in the top-right corner of the **Theme** preview area.
2.  Select **CheckButton** in the upcoming pop-up menu. By doing this, you will see a list of this component’s relevant properties separated by tabs on the right-hand side of the theme preview.
3.  Switch to the fourth tab, which looks like a polaroid icon with a mountain in it. Press the plus (**+**) icons for the **off** and **on** properties.
4.  From the `dark-ui-checkbutton-off.png` to the *off* slot and, similarly, drag `dark-ui-checkbutton-on.png` to the *on* slot.
5.  Interact with **CheckButton** in the theme’s preview.

This will effectively change the look of **CheckButton**. Your **Theme** panel will look as follows:

![Figure 11.10 – We have changed the look of the CheckButton component with custom assets ](img/Figure_11.10_B17473.jpg)

Figure 11.10 – We have changed the look of the CheckButton component with custom assets

**CheckButton** is a simple component with two main states: on and off. We were not interested in altering the disabled versions of its two states, simply because the UI kit does not have the assets for that permutation. If you think you’ll never have this component in a disabled state, then you don’t have to create and assign art either.

Let’s attack a different component this time. Although its name is similar, and it comes with states similar to **CheckButton**, a somewhat disguised property of this node makes it function as two distinct components. Enter **CheckBox**.

## Changing a CheckBox and discovering radio buttons

This is going to be a similar effort, but we’ll utilize more assets and fill out more properties. Let’s keep the momentum going and add a new item to the theme:

1.  Using the plus (**+**) icon button again, choose **CheckBox** from the upcoming item list.
2.  The fourth tab may still be active. If not, switch to it and do the following:
    1.  Assign `dark-ui-checkbox-off.png` to the **unchecked** property.
    2.  Assign `dark-ui-checkbox-on.png` to the **checked** property.
    3.  Assign `dark-ui-radio-off.png` to the **radio_unchecked** property.
    4.  Assign `dark-ui-radio-on.png` to the **radio_checked** property.

When you prepare your assets, pick filenames that are close enough to the state the assets will be assigned to. So, associating these files between the **FileSystem** and **Theme** panels would feel easy. After making these changes, this is what we have:

![Figure 11.11 – CheckBox is the latest item we have customized for our Dark theme ](img/Figure_11.11_B17473.jpg)

Figure 11.11 – CheckBox is the latest item we have customized for our Dark theme

The preview area has the **CheckBox** component for you to test, but no radio button. There is no **RadioButton** component in Godot. Despite adding the assets for it, we can’t simulate it yet. Nevertheless, we can tweak a **CheckBox** component so that it acts like a radio button.

Since we need to physically place a `UI.tscn` scene:

1.  Turn the visibility off for the **Panel** node by clicking its eye icon in the **Scene** panel.
2.  Select the root node, then add an **HBoxContainer** node. Select this new node right away so that you can do the following:
    1.  Add a **VBoxContainer**, **VSeparator**, and another **VBoxContainer** to it.
    2.  Add two **CheckBox** nodes inside these two **VBoxContainer** nodes.
    3.  For the first two `Multiple Choice 1` and `Multiple Choice 2`, respectively.
    4.  For the last two `Single Choice 1` and `Single Choice 2`, respectively.

We’re not done yet, but the following screenshot shows what’s happened so far:

![Figure 11.12 – Four checkboxes organized in a questionnaire fashion ](img/Figure_11.12_B17473.jpg)

Figure 11.12 – Four checkboxes organized in a questionnaire fashion

We are a few steps closer to turning two of those checkboxes into radio buttons – specifically, the last two since we gave them some text that mentions a single choice. Thus, while you have **CheckBox2** in the **VboxContainer2** node selected, do the following:

1.  Assign a **New ButtonGroup** to its **Group** property in the **Inspector** panel.
2.  Click the down arrow in that **Group** slot to expand a dropdown menu and select **Copy**.
3.  Select the **Checkbox** node in **VBoxContainer2** and choose the **Paste** option by expanding its **Group** options. This will link the two checkboxes because they will be sharing the same button group.

You should notice a drastic change between the two sets of checkboxes. Whereas the first two still look like checkboxes, the last two have circular icons next to them, as shown in the following screenshot:

![Figure 11.13 – Two checkboxes have been converted into radio buttons ](img/Figure_11.13_B17473.jpg)

Figure 11.13 – Two checkboxes have been converted into radio buttons

By sharing the same button group, checkboxes turn into radio buttons. In this exercise, it was sufficient to create and assign a generic **ButtonGroup** object. However, if you want to have a group of radio buttons in one area of your application, then another collection somewhere else that governs a different set of radio buttons, you may have to create named **ButtonGroup** objects and assign them accordingly.

We won’t cover that kind of scenario since we seem to be missing something more important that we have wanted for a while. Neither the checkboxes nor the radio buttons we worked so hard for are reflecting the artistic direction we defined in our theme. Let’s see how we can utilize our theme.

## Attaching a theme

Previously, we mentioned that using themes would help you style components faster. It’s true, but we haven’t tested this claim yet. Since we’ve already prepared the styles for the checkboxes and radio buttons, all there is left to do is assign the theme to these components:

1.  Select the **HBoxContainer** node in the **Scene** panel and expand the **Theme** section in the **Inspector** panel.
2.  Drag `Dark.tres` from **FileSystem** to fill the empty **Theme** slot.

There you have it! We didn’t even have to select each component and assign the themes one by one. A higher-level structure such as **HBoxContainer** was enough to assign the theme to so that its children could use the relevant parts.

Do you see the real potential here? Assigning a theme to a root element will be enough most of the time. That being said, since each component can be assigned its own theme, but it doesn’t have to, you can have all sorts of permutations. In its simplest form, assigning a theme to a root node will be enough in most scenarios.

So far, we’ve been styling relatively simple UI nodes, such as **CheckButton** and **CheckBox**. Maybe we could tackle another node that has a few moving parts, such as a **VSlider**.

## Altering a vertical slider component

A vertical slider component, **VSlider**, is useful when you want to give your players an easy way to adjust the ratio or quantity of things, such as tradeable items during a game session, music volume, or the brightness level in the game’s settings. Likewise, you can use an **HSlider** node, which is the horizontal version, but both accomplish similar tasks.

Since we only have the graphic assets for a **VSlider**, we’ll only cover this styling. If you desire, it’s possible to convert the existing assets that are compatible with an **HSlider**. You’ll have to rotate each part 90 degrees and save them accordingly. To do so, you must follow these steps:

1.  Add **VSeparator** and **VSlider** nodes to **HBoxContainer** in the **Scene** panel.
2.  Using the `75` for the **Value** property for **VSlider**.
3.  Double-click `Dark.tres` in **FileSystem** to bring up its details. Add **VSlider** as a new type using the good old button with the plus (**+**) icon.
4.  Activate the fourth tab in this new type’s custom properties and assign `dark-ui-vslider-grabber.png` to both **grabber** and **grabber_highlight**.
5.  Switch to the fifth tab, which looks like a square rainbow.
6.  Attach a `dark-ui-vslider-grabber-area.png` to the **Texture** property.
7.  Expand the `6` for the **Bottom** property.

*   Bring up the theme preview again by double-clicking `Dark.tres` or switching to the **Theme** panel at the bottom.*   Instead of repeating the same effort for the **grabber_area_highlight** property, click the plus (**+**) button near its slot, then grab and drop the **grabber_area** property’s style onto the **grabber_area_highlight** slot. Alternatively, you can copy the slot from **grabber_area** and paste it into **grabber_area_highlight** using the dropdown menus.*   Attach a `dark-ui-vslider-slider.png` to the **Texture** property.*   Expand the `6` for the **Bottom** and **Top** properties.*   Make the `1` for all its properties.*   Press *F6* and admire your hard work.

We took many steps here, but there were only one or two new things. First, we repurposed one of the styles to be used for a different property by dragging and dropping it. This is a shortcut method instead of copying and pasting between slots. It’s useful when both slots are near each other. If you are copying elements where the slots are on different panels, then you still have to resort to the copy and paste method in dropdown menus.

Secondly, we adjusted a different type of margin, **Expand Margin**. The slider has two separate parts that constitute its track where the scrolling occurs, so we had to adjust this special margin so that it fits the blue part inside the outer part. Take a look at the following screenshot; you will see that there is a blue filler under the grabber inside the track of **VSlider**:

![Figure 11.14 – It took a few more steps but the VSlider component has been thematized ](img/Figure_11.14_B17473.jpg)

Figure 11.14 – It took a few more steps but the VSlider component has been thematized

It’s easier to see the effect live than reading it. So, when you launch the `UI.tscn` scene, try to interact with the grabber and see how the component fills its track with blue, depending on the position of the grabber.

## Wrapping up

This concludes our work in setting up a theme. Although we have styled only a handful of nodes, feel free to practice with the rest of the same UI kit or pick another one from the website to try it on other **Control** nodes.

All in all, working with themes or individually styling components entails two things. Primarily, you can either assign textures directly to some of the properties or indirectly into the appropriate slot by creating a **StyleBoxTexture**. Secondly, there are some numerical properties you can tweak. We haven’t covered this latter case. For example, you can adjust the line height of components that deal with text rendering. These cases are easy to comprehend and test. So, we opted to show you more head-scratching cases.

Hopefully, by practicing what we have shown so far and discovering more on your own, you will be able to apply beautiful graphic designs to your game.

# Summary

We started this chapter by debating what UIs are. We did this via a brief philosophical and theoretical explanation.

Assuming your games will require UIs, we investigated a practical use case such as constructing a note left by Clara’s uncle. This work necessitated us to work with multiple **Control** nodes – that is, the **Button**, **Panel**, and **Label** nodes.

During this effort, not only did we employ the components we needed, but we also styled them to match a specific artistic style.

For the sake of not repeating ourselves and taking the styling to the next level, we presented how using themes might be a time saver. To that end, we showed you how to utilize UI kits you could find online by assigning these kits’ individually exported graphics assets to the properties of **Control** nodes.

UIs are, in a way, a tool for the player to interact with the game. That being said, in the next chapter, we’ll discover a more direct way to interact with the game world without the help of UIs.

# Further reading

In the introduction, we talked about when a UI is necessary. However, there are situations when the best interface is no interface at all. There is an app – sorry, a book – for that by *Golden Krishna*: *The Best Interface Is No Interface: The simple path to brilliant technology*. He talks about how introducing more steps and elements disguised as a UI is nothing but interference.

We’ve already discussed the possibility of having games without a UI, but we’ll rest that argument for now. It might be better for you to be exposed to as much information and examples as possible at this point. So, the following are a few technical and practical resources:

*   [https://www.toptal.com/designers/gui/game-ui](https://www.toptal.com/designers/gui/game-ui)
*   [https://webdesign.tutsplus.com/articles/figma-ui-kits-for-designers--cms-35706](https://webdesign.tutsplus.com/articles/figma-ui-kits-for-designers--cms-35706)
*   [https://ilikeinterfaces.com/](https://ilikeinterfaces.com/)
*   [https://www.gameuidatabase.com/](https://www.gameuidatabase.com/)

This chapter also showed you how to assign fonts to components. There are a lot of freely available fonts out there but be careful and read their licenses. They might be downloadable but some of them can’t be used in commercial work. The same kind of warning goes for anything else too, especially graphics assets.