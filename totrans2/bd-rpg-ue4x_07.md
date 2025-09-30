# Chapter 7. Gold, Items, and a Shop

Now that you have created an NPC that talks to the player, it is time to allow the NPC to help the player. In this chapter, we will use the NPC as a shop owner, displaying items for the user to buy. Before we do this, the user is going to need some sort of currency to buy the items. We will cover the following topics in this chapter:

*   Setting and getting gold instances
*   Item data
*   The shop screen framework
*   The item button framework
*   Linking the item data

# Setting and getting gold instances

While we move on to making a shopping interface, via the **Shop** button, we must first be able to pull the currency in order to pay for items in the shop. In a previous chapter, we discussed and made placeholders for gold, but we did not actually create gold values. In this game, we would like gold to be dropped by enemies at the end of battle. In this case, enemies will need some sort of gold data that we can add to the player's gold data (eventually, items will need this gold data that is tied to them as well). In [Chapter 4](ch04.html "Chapter 4. Pause Menu Framework"), *Pause Menu Framework*, we created a pause menu that has a gold placeholder, and we will now add gold to this pause menu.

First, let's add a `Gold` property to `FEnemyInfo.h`. Navigate to **Source** | **RPG** | **Data**, open `FEnemyInfo.h`, and add a `Gold` property of an integer data type to your `EnemyInfo` table, as follows:

[PRE0]

We now need to tie the `Gold` property with our standard `GameCharacter` properties so that we can update any instance of an enemy with the proper gold value. Next, you will open `GameCharacter.h`, which is located in **RPG** under **Source**, and add a public `UProperty` to the `UCharacter` class for gold similar to that in `FEnemyInfo.h`:

[PRE1]

Then, head into `GameCharacter.cpp` to set the return value of the gold that is equal to the value set in `EnemyInfo`, so that each instance of this particular enemy will return the amount of gold set in the enemy's data table:

[PRE2]

When you are finished, the enemy's character information in `GameCharacter.cpp` will look like this:

[PRE3]

We now need to choose when to accumulate the gold, and in this case, we will accumulate the gold from combat. So, navigate to **Source** | **RPG** | **Combat**, open `CombatEngine.h`, and create a public gold variable that we will use to store all the gold won in the battle:

[PRE4]

When you have finished declaring the `GoldTotal` variable, the `CombatEngine.h` file will look like this:

[PRE5]

The next step that we need to perform is telling the engine when to give the gold to the player. As mentioned earlier, we want players to win gold from enemies that can easily be integrated into our combat engine. Navigate to **Source** | **RPG** | **Combat**, and open `CombatEngine.cpp`. Let's first scroll down to the `for` loop that we created in [Chapter 3](ch03.html "Chapter 3. Exploration and Combat"), *Exploration and Combat*, to check for a victory. Just above this `for` loop, declare a new `Gold` integer, and set it to `0`:

[PRE6]

This will assure that, if we don't have a victory and need to cycle through the `for` loop again, the gold gained in battle will reset to 0\. Next, we need to accumulate the gold from every enemy killed; thus, within the `for` loop, we have `Gold` increment by each enemy's gold:

[PRE7]

Your `for` loop will now look like this:

[PRE8]

After the `for` loop, you will still have an `if` condition that checks whether the enemy party is dead; if the enemy party is dead, the combat phase will change to the victory phase. If the condition is `true`, it means that we won the battle; therefore, we should be rewarded with the gold from the `for` loop. Since the `Gold` variable that we want to add is in the `GoldTotal` variable, we simply set the local `Gold` variable to the new value of `GoldTotal`:

[PRE9]

When you are finished, your `if` condition will now look like this:

[PRE10]

Now that we have set enemies to drop gold after the player is victorious in battle, the next thing that we need to do is add gold to our game data; more specifically, it would be best to add it in `RPGGameInstance.h`, since an instance of the game will always be active. It would be unwise to add the gold data to a party member unless there is a specific party member who will always be in the game. So, let's open `RPGGameInstance.h` located in **RPG** under **Source**.

As a public property, add another integer to `Game Data` that we will call `GameGold`. Also, ensure that `GameGold` is read- and write-enabled because we want to be able to add and subtract gold; therefore editing of `GameGold` must be enabled:

[PRE11]

Now that we can create instances of `GameGold`, go to your `RPGGameMode.cpp` file where you originally set up the logic for the game over and victory conditions; in the victory condition, create a pointer to `URPGGameInstance` that we will call `gameInstance`, and set it equal to a cast to `GetGameInstance`:

[PRE12]

We can now use `gameInstance` to add the total gold that we got from the battle to `GameGold`:

[PRE13]

At this point, the value of `GameGold` that we are using as the player's gold will now be incremented by the gold won in the battle. The `tick` function in `RPGGameMode.cpp` will now look like this:

[PRE14]

Now, you need to make sure that all your changes are saved and recompile your entire project (you may need to restart UE4).

We can now adjust the gold value of each enemy character that we have from the enemy's Data Table. In **Content Browser**, navigate to the **Enemies** Data Table located at **Data** under **Content**. In the Data Table, you will now see a **Gold** row. Add any value that you want to the **Gold** row, and save the Data Table:

![Setting and getting gold instances](img/B04548_07_01.jpg)

Now that an enemy has a gold value, there is a real value that is bound to the `Gold` variable in `EnemyInfo` that gets added to `GameGold` if the player is victorious in battle. However, we need to display that gold; luckily, we still have a placeholder for the gold in our pause menu. Open the **Pause_Main** Widget Blueprint, and click on the **Editable_Gold** Text Block that we created in [Chapter 4](ch04.html "Chapter 4. Pause Menu Framework"), *Pause Menu Framework*. In the **Details** panel, go to **Content** and create a binding for the Text Block, which will open the graph for **Get Editable Gold Text**:

![Setting and getting gold instances](img/B04548_07_02.jpg)

The first thing that we need to do is get the game instance of **RPGGameInstance** by creating a **Get Game Instance** function located under **Game** and setting it as an object of **Cast To RPGGameInstance**:

![Setting and getting gold instances](img/B04548_07_03.jpg)

We can then get the `GameGold` variable from **RPGGameInstance**, which is the variable that stores the current gold total for the game. It is located in **Game Data** under **Variables**. Link it to the **As RPGGameInstance** pin in **Cast To RPGGameInstance**:

![Setting and getting gold instances](img/B04548_07_04.jpg)

Lastly, link **Game Gold** to **Return Value** in **ReturnNode** and allow **Get Editable Gold Text** to trigger **Cast To RPGGameInstance**, which will trigger **ReturnNode**. Your **Get Editable Gold Text** binding will now look like this:

![Setting and getting gold instances](img/B04548_07_05.jpg)

If you test this now, you will be able to get into battle, win gold from your enemies on victory, and now you will be able to see your gold accumulate in your pause menu. We can use these same variables to add to any menu system, including a shop.

# Item data

Now that we are finished with the gold creation, we need to create one more thing before we make a shop, that is, items. There are many ways to make items, but it is best to keep an inventory and stats of items through the use of Data Tables. So, let's first create a new C++ `FTableRowBase` struct similar to the `CharacterInfo` structs that you previously created. Our files will be called `ItemsData.h` and `ItemsData.cpp`, and we will put these files where our other data is; that is, by navigating to **Source** | **RPG** | **Data**. The `ItemsData.cpp` source file will include the following two header files:

[PRE15]

The `ItemsData.h` header file will contain definitions of all the item data that we will need. In this case, the item data will be stats that the player has, since items will most likely affect stats. The stats only need to be of the integer type and read-enabled since we won't be changing the value of any of the items directly. Your `ItemsData.h` file will look something like this:

[PRE16]

At this point, you can recompile, and you are now ready to create your own Data Table. Since we are creating a shop, let's create a Data Table for the shop in **Content Browser** and in the `Data` folder by navigating to **Miscellaneous** | **Data Table**, and then using **Items Data** as the structure.

![Item data](img/B04548_07_06.jpg)

Name your new Data Table **Items_Shop**, and then open the Data Table. Here, you can add as many items as you want with whatever kinds of stat you would like using the **Row Editor** tab. To make an item, first click on the **Add** button in **Row Editor** to add a new row. Then, click on the textbox next to **Rename** and type in **Potion**. You will see that you have a potion item with all the other stats zeroed out:

![Item data](img/B04548_07_07.jpg)

Next, give it some values. I will make this a healing potion; therefore, I will give it an **HP** value of **50** and a **Gold** value of **10**.

![Item data](img/B04548_07_08.jpg)

The purpose of this Data Table is also to store every item that our shop owner will carry. So, feel free to add more items to this Data Table:

![Item data](img/B04548_07_09.jpg)

# The shop screen framework

Now that we are done with creating items, we can move on to creating the shop. In the previous chapter, we created dialog boxes for our shop owner, and in one of the dialog boxes, we created a **Shop** button that, when clicked, will open up a shop menu. Let's create this shop menu by first creating a new Widget Blueprint by navigating to **Content** | **Blueprints** | **UI** | **NPC**. We will call this Widget Blueprint **Shop** and open it:

![The shop screen framework](img/B04548_07_10.jpg)

We will make the shop in a similar format to that of our pause menu, but we will keep it simple because all we need for now is a Scroll Box that will hold the shop's items, as well as an area for gold, and an **Exit** button.

To expedite this process, you can simply copy and paste the elements from your existing menu systems that you wish to reuse into the **Shop** Widget Blueprint. We can do this by navigating to **Content** | **Blueprints** | **UI** and opening the **Pause_Main** and **Pause_Inventory** Widget Blueprints, which we created in the previous chapters. From **Pause_Main**, we can copy the **Menu_Gold**, **Editable_Gold**, **Button_Exit**, **Menu_Exit**, and **BG_Color**, and paste them into the **Shop** Widget Blueprint.

We can also copy the **ScrollBox_Inventory** and **Title_Inventory** from the **Pause_Inventory** Widget Blueprint and paste them into the **Shop** Widget Blueprint. When you are done, your **Shop** Widget Blueprint will look like this:

![The shop screen framework](img/B04548_07_11.jpg)

Here, edit the **Shop** Widget Blueprint so that the title reads as **Shop** instead of **Inventory**:

![The shop screen framework](img/B04548_07_12.jpg)

You will now need to link the **Shop** Widget Blueprint to the Shop button in the **Shop_Welcome** Widget Blueprint. To do this, open the **Shop_Welcome** Widget Blueprint by navigating to **Content** | **Blueprints** | **UI** | **NPC**, select **Button_Shop**, and then click on the **+** button to the right of the **OnClicked** event by navigating to **Details** | **Events**:

![The shop screen framework](img/B04548_07_13.jpg)

This will automatically open the graph with a newly created **OnClicked** event for **Button_Shop**:

![The shop screen framework](img/B04548_07_14.jpg)

Here, you can simply mimic the same actions you used to open the dialog boxes when the player clicks on the **Talk** button. The only difference is that, instead of creating a new **Shop_Talk** widget, the **Shop** widget will create the **Create Shop Widget** for you. The graph for **Button_Shop** will look like the following screenshot:

![The shop screen framework](img/B04548_07_15.jpg)

You will now be able to test the shop by talking to the NPC and clicking on the **Shop** button, which will now open the shop:

![The shop screen framework](img/B04548_07_16.jpg)

You will notice that nothing is yet visible in the shop, not even the gold. To display the gold on the screen, you need to repeat the steps you performed earlier in this chapter when you displayed the gold in the **Pause_Main** Widget Blueprint. But this time, open the graph in the **Shop** Widget Blueprint, and then create a binding for the **Editable_Gold** Text Block by navigating to **Details** | **Context**:

![The shop screen framework](img/B04548_07_17.jpg)

Your graph will automatically open, and you will notice a **Get Editable Gold Text** function with a **ReturnNode**. Since you will be getting the gold from the same game instance that you did when getting the gold from the **Pause_Main** Widget Blueprint, you can simply copy and paste all the nodes from the **Get Editable Gold Text** function into **Pause_Main**, and link them to the **Get Editable Text** function in the **Shop** Widget Blueprint. When you are done, the **Get Editable Gold Text** function in the **Shop** Widget Blueprint will look like this:

![The shop screen framework](img/B04548_07_18.jpg)

Next, we will create the **Button_Exit** functionality in the **Shop** Widget Blueprint by creating an **OnClicked** event (by navigating to **Details** | **Events**) for **Button_Exit**:

![The shop screen framework](img/B04548_07_19.jpg)

When the graph opens, link the **OnClicked** event to the **Remove from Parent** function:

![The shop screen framework](img/B04548_07_20.jpg)

At this point, when you test the shop, you will see the gold and be able to exit the shop screen.

# The item button framework

Before we link our items to the shop, we will first need to create a framework in which the items are placed in the shop. What we would like to do is create a button for each item that the shop owner sells but, in order to make the interface scalable in such a way that NPCs can hold different selectable items, it would be wise to create a Scroll Box framework that holds a single button with a default value for the item's text/description. We can then dynamically draw the button for as many items as the shop owner carries, as well as dynamically draw the text on each button.

To do this, we must first create a Widget Blueprint by navigating to **Content** | **Blueprints** | **UI** and call it **Item**:

![The item button framework](img/B04548_07_21.jpg)

Open **Item**. Since we are going to make the items clickable, we will program a button. To make the button, all that we will need is the button itself and text for the button; we will not need a Canvas Panel because we will eventually be adding this button to the Scroll Box of our shop. So, from the **Hierarchy** tab, delete the Canvas Panel, and drag a button from **Palette**. We will name this button, **Button_Item**:

![The item button framework](img/B04548_07_24.jpg)

Finally, we will place a Text Block in the button that we just created and name it **TextBlock_Item**:

![The item button framework](img/B04548_07_25.jpg)

Once done, navigate to **Details** | **Content**, and create a binding for the text in the Text Block. This will automatically open the graph with a **Get Text** function:

![The item button framework](img/B04548_07_26.jpg)

Create a new **Item** variable of the **Text** type:

![The item button framework](img/B04548_07_27.jpg)

Drag the **Item** variable into the graph, select **Get** to drop in a getter for the **Item** variable, and then link it to the **Return Value** pin of **ReturnNode**:

![The item button framework](img/B04548_07_28.jpg)

# Linking the item data

It is now time to link the item data that we created at the beginning of this chapter to the shop using the **Item** button framework we just created. To do this, we will add a functionality to display every item in our **Items_Shop** Data Table using the **Item** button framework that we created in the previous section. First, open **Event Graph** in the **Shop** Widget Blueprint. Link the **Get Data Table Row Names** function located in Data Tables to **Event Construct**:

![Linking the item data](img/B04548_07_29.jpg)

Then, from the **Select Asset** drop-down menu, select **Items_Shop**:

![Linking the item data](img/B04548_07_30.jpg)

This will get the names of every item in the **Items_Shop** Data table that we created earlier in this chapter. Here, we need to create an instance of the **Item** Widget Blueprint for every item row. This will create a button for every item with the correct corresponding item name. To do this, create a **ForEachLoop** located at **Array** under **Utilities** and allow the **Get Data Table Row Names** function to execute it. Link the **Out Row Names** pin to the **Array** pin of the **ForEachLoop** so that every row in the Data Table becomes an element of the array in the **ForEachLoop**:

![Linking the item data](img/B04548_07_31.jpg)

Next, we need to loop through each element of the array of row names and, for each row, we need to create a new instance of the **Item** Widget Blueprint. To do this, link the **Create Item Widget** action located under **User Interface** to the **Loop Body** pin in the **ForEachLoop**. Let the class instance be **Item** that can be selected from the **Class** drop-down menu in the **Create Item Widget** action:

![Linking the item data](img/B04548_07_41.jpg)

Then, for every item created, set the **Item** variable that is created for every **Item** widget instance to the value of each element in the array. You can create the **Set Item** action, by right-clicking anywhere in **Event Graph**, unchecking **Context Sensitive**, and locating **Set Item** by navigating to **Class** | **Item**:

![Linking the item data](img/B04548_07_33.jpg)

**Create Item Widget** can now launch **Set Item**, and set the **Return Value** pin value of **Create Item Widget** to the **Target** pin value of **Item**:

![Linking the item data](img/B04548_07_32.jpg)

At this point, we have not yet set the element of the array to the item that we set in the **Item** widget; so, to do this, we can simply link the **Array Element** pin from the **ForEachLoop** to the **Item** pin in the **Set Item** action:

![Linking the item data](img/B04548_07_36.jpg)

Lastly, we are going to have our Scroll Box that we created in the **Shop** Widget Blueprint hold all of our item instances. To do this, after we set each item to the correct name, we will add the item instance as a child to the **ScrollBox_Inventory** Scroll Box that we created earlier in this chapter. This is done by simply calling the **Add Child** function located in **Panel** under **Widget** after we set the item:

![Linking the item data](img/B04548_07_37.jpg)

Then, we set the **Content** value of the child to the **Return Value** pin of the item:

![Linking the item data](img/B04548_07_38.jpg)

Lastly, the **Target** pin of the child needs to be linked to **ScrollBox_Inventory**, which can be dragged into your **Event Graph** from **Variables**. If you do not see the **ScrollBox_Inventory** variable in your variables, go back to the **Designer View**, select the **ScrollBox_Inventory**, and make sure **is variable** is checked:

![Linking the item data](img/B04548_07_39.jpg)

At this point, if you test your shop, you will see the shop populated with every item listed in your Data Table:

![Linking the item data](img/B04548_07_40.jpg)

You will be able to add even more items to your Data Table and these items will automatically appear in your shop.

# Summary

In this chapter, we created a currency system for our game along with the ability for our enemies to drop gold. We also created a new set of data that contains items and their stats, and we have now populated the shop owner's store to display the items currently for sale in the shop.

In the next chapter, we will add the buying functionality to the shop along with the usage of an item and consumption.