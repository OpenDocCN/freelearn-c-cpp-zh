# Chapter 8. Inventory Population and Item Use

In the previous chapter, we learned how to add a shop, which holds items. In this chapter, we will go a step further by allowing a user to buy items from the shop and use those bought items in their dynamically populated inventory screen. Once done, we will use similar ideas to equip items to party members that will be used to increase the stats of the wearer.

By the end of this chapter, we will learn how to create logic in our **Shop** Widget Blueprint that populates the inventory Scroll Box in the **Shop** Widget Blueprint with the buttons created through the **Item** Data table from the **Item** Widget Blueprint. Now that we have the logic set up, we need to allow the user to interact with the buttons by being able to buy any item that they click on in the shop, so long as they have enough money. Since the issuer interacts with the dynamically populating buttons, it is important that we have our logic executed when the user presses a button, which is located in the **Item** Widget Blueprint.

If you have any other Blueprints in your **Event Graph** that you may have put together earlier, you can ignore them since an interaction will allow us to start over using some different methodologies.

First, we must note that the **Item** Blueprint will house logic that should happen anytime a button from that Blueprint is clicked. So, at the moment, we are planning to have the button populate the shop, but in the player's inventory, the logic would need to be different, depending on which screen we are on. This means that we will first need to find out which screen the player is on, and then fire off a series of actions based on the screen they are on. It would be easy to do this with Booleans from the **OnClicked** event, which will check to see which menu the player is in and branch off different logic, depending on which menu the player is currently in.

Since we are concerned with the difference between the behavior of the buttons in the **Pause_Inventory** screen versus the **Shop** screen, we must first create a Boolean that will stay active throughout the life of the character. In this case, we will use the Field Player to hold our important item variables.

In this chapter, we will cover the following topics:

*   Creating the Field Player Booleans
*   Determining whether the inventory screen is on or off
*   Logical difference between inventory and shop items
*   Finishing the inventory screen
*   Using the items

# Creating the FieldPlayer Booleans

Go to the Field Player by navigating to **Content** | **Blueprints** | **Characters**, and select **FieldPlayer**. Open **FieldPlayer** and navigate to **Event Graph**:

![Creating the FieldPlayer Booleans](img/B04548_08_01.jpg)

Here, we create a new variable under the **Blueprint** tab by navigating to **+Add New** | **Variable**. Next, we create a new `inventoryScreen` Boolean. Then, we need to make the variable public. This Boolean will be responsible for keeping true or false values, depending on whether the player is on the inventory screen. We may need more variables like these in the figure, but for now, we will just use this variable:

![Creating the FieldPlayer Booleans](img/B04548_08_59.jpg)

When you are finished creating the `inventoryScreen` variable, compile the Blueprint.

# Determining whether the inventory screen is on or off

We will now set the `inventoryScreen` variable in its proper place. The best place to put this is when the inventory menu pops up. So, go to **Pause_Inventory** by navigating to **Content** | **Blueprints** | **UI**. In **Pause_Inventory**, locate the **Event Construct** in the Event Graph (if an **Event Construct** does not exist yet, create one), and from here, get every actor from the Field Player class by creating **Get All Actors of Class**, which is located under **Utilities** in the **Actions** menu:

![Determining whether the inventory screen is on or off](img/B04548_08_03.jpg)

Under **Actor Class** in the **Get All Actors Of Class** function, change the actor to **Field Player**:

![Determining whether the inventory screen is on or off](img/B04548_08_04.jpg)

From the **Out Actors** pin, in the **Get All Actors Of Class** function, you will need to attach a **GET** function. This will take an array of all actors in your Field Player class and allow access to individual members of the class:

![Determining whether the inventory screen is on or off](img/B04548_08_05.jpg)

Lastly, open your all possible actions and uncheck **Context Sensitive**. Go to **Set Inventory Screen** by navigating to **Class** | **Field Player**:

![Determining whether the inventory screen is on or off](img/B04548_08_06.jpg)

Once done, connect the **Target** pin of your **Set Inventory Screen** to the right-hand side pin of **GET**. Also, make sure that the **Inventory Screen** is checked, which means that we set the **Inventory Screen** to true here. At this point, you can also link **Event Construct** to fire off **Get All Actors Of Class**, which will activate the **Set Inventory Screen**:

![Determining whether the inventory screen is on or off](img/B04548_08_07.jpg)

We will also need to make sure that the Boolean is set to false when the player leaves the inventory screen, so clone another **Set Inventory Screen** Boolean, and set it to false. Link the **Target** pin back to the **GET** from **Get All Actors Of Class**, and activate it when the inventory window closes:

![Determining whether the inventory screen is on or off](img/B04548_08_08.jpg)

We will later come back to **Pause_Inventory** to add the button population logic, similar to the shop in the previous chapter. However, now that we have our Booleans set, we will be able to tell whether the player is viewing the inventory or is navigating the shop (if the Boolean is false).

# Logical difference between the inventory and shop items

Let's now open the **Item** Widget Blueprint by navigating to **Content** | **Blueprints** | **UI**:

![Logical difference between the inventory and shop items](img/B04548_08_09.jpg)

At this point, we should not have any logic for the button, which is necessary because it gives us the actions that the button will perform in conjunction with the game. To add a functionality to the button, click on the button, navigate to **Details** | **Events** | **OnClicked**, and then click on **+**:

![Logical difference between the inventory and shop items](img/B04548_08_10.jpg)

Here, we will need to do a few things. Firstly, we know that this Blueprint will be responsible for all of the button mechanics regarding both the shops and the character's inventory and the mechanics will be different since the character buys items from the shop and uses items from the inventory. Since these different game screens provide different actions, it would be wise to first check whether a user is in the shop or in their inventory. To do this, we should first bring in the **Get All Actors Of Class** function, and get all the actors from the Field Player class. Then, we need to link the **Out Actors** pin to **GET**. Finally, have the **OnClicked** event fire off **Get All Actors Of Class**:

![Logical difference between the inventory and shop items](img/B04548_08_11.jpg)

At this point, we can open our **Actions** window, and go to the **Get Inventory Screen** by navigating to **Class** | **Field Player**. You will need to uncheck **Context Sensitive** to see this option:

![Logical difference between the inventory and shop items](img/B04548_08_12.jpg)

You will then link the **Target** pin of the **Inventory Screen** node to the blue **GET** pin. This will allow us to access the **Inventory Screen** Boolean from the Field Player class:

![Logical difference between the inventory and shop items](img/B04548_08_13.jpg)

It is now time to create a branching system that will perform logic, depending on whether the player is shopping or whether they are in their inventory. We will use our **Inventory Screen** Boolean for this. Let's first bring in a branch by navigating to **Utilities** | **Flow Control** in the **Actions** menu:

![Logical difference between the inventory and shop items](img/B04548_08_14.jpg)

Here, we link the condition of your branch to the **Inventory Screen** condition. Then, have the **Get All Actions Of Class** function activate the branch. At this point, when the player clicks on the button, we will check to see whether the **Inventory Screen** is true (or if the player is on the inventory screen). If they are not on the inventory screen, then it means that the player is on some other screen; in our case, the shop:

![Logical difference between the inventory and shop items](img/B04548_08_15.jpg)

Before we continue with the rest of the **Item** button logic, we need to think about our logical flow. If the user is in the shop, and the user clicks on an item to be purchased, then if that person has enough money to purchase the item, the item should be placed into some sort of a collection or array that can populate the user's inventory screen. Because of this mechanic, we will need to seek some sort of global array that will be able to hold an array of items that the player has purchased. To do this, go to the **FieldPlayer** Event Graph and add a new text array named **arrayItem**. Also, make sure that this variable is set to public and is editable:

![Logical difference between the inventory and shop items](img/B04548_08_16.jpg)

# Finishing the inventory screen

Navigate to the **Pause_Inventory** Event Graph. While **Context Sensitive** is off, bring in the **Get Array Item** from the **Actions** window by navigating to **Class** | **Field Player**:

![Finishing the inventory screen](img/B04548_08_17.jpg)

Once done, connect the **Target** pin of **Array Item** to **GET** so that you can get every item that is sent to that array once we populate the array in the **Items** Blueprint:

![Finishing the inventory screen](img/B04548_08_18.jpg)

Now that we have the array of items in the player's inventory, we will now loop through each element, and create an item from every element in the array. To do this, create a **ForEachLoop** by navigating to **Utilities** | **Array**. Link **Array Item** from your **arrayItem** variable to the **Array** tab in the **ForEachLoop**. Then, have **SET Inventory Screen** activate the **ForEachLoop**:

![Finishing the inventory screen](img/B04548_08_19.jpg)

Just like what we did when populating the buttons for the shop, we would want this `for` loop to be responsible for adding buttons from the **Items** Widget Blueprint. So, in the body of the `for` loop, we need to create the **Item** widget by first navigating to **User Interface** | **Create Widget** in the **Actions** window:

![Finishing the inventory screen](img/B04548_08_20.jpg)

Then, we need to change the **Class** dropdown to **Item**, and link it to **Loop Body** in the **ForEachLoop**:

![Finishing the inventory screen](img/B04548_08_22.jpg)

You will then need to set the text for each element in the array. So, open the **Actions** window and with **Context Sensitive** off, bring in **Set Item** by navigating to **Class** | **Item**.

Link the **Item** pin to the **Array Element** pin from the **ForEachLoop**. Then, set the **Target** pin of **Set Item** to the **Return Value** of **Create Item Widget** and have **Create Item Widget** activate the **Set Item**:

![Finishing the inventory screen](img/B04548_08_23.jpg)

Lastly, we will need to add the **Item** widget to the Scroll Box that we created in **Pause_Inventory**. Simply create an **Add Child** node that is located at **Panels** under **Widget**. Then, link **ScrollBox_Inventory** from your variables to the **Target** pin of **Add Child** (if you do not see **ScrollBox_Inventory** as a default variable, make sure you go back into the Designer View of **Pause_Inventory**, select the **ScrollBox_Inventory**, and check **is variable**, then have the **Content** pin of **Add Child** be the **Return Value** of **Create Item Widget**). Finally, have the **Set Item** node start up the **Add Child** node:

![Finishing the inventory screen](img/B04548_08_24.jpg)

When you are done, your **Pause_Inventory** Blueprint will look like this:

![Finishing the inventory screen](img/B04548_08_25.jpg)

# Buying items

Head back into the **Item** Blueprint. Where we left off before, we allowed that upon clicking a button, we would get all actors from the Field Player class. Here, we set up a branch that checks whether the **Inventory Screen** Boolean is true or false (which means that we check whether the player is on the inventory screen; if they are not on the inventory screen, we will perform the buying logic in our shop).

Let's first bring in a **Get Data Table Row** function located under **Utilities** in the **Actions** window:

![Buying items](img/B04548_08_59.jpg)

Then, set the Data Table to **Items_Shop**. This will allow us to get every row from the **Items_Shop** Data Table. Then, link the **False** pin from the branch that we created to the execution of **Get Data Table Row**:

![Buying items](img/B04548_08_26.jpg)

You may have noticed that we can select any row name from the Data Table. In this case, we just need to get the row name of the item that is currently selected. To do this, bring in **Get** of the **Item** text variable that you created in the previous chapter in this class. You need to link it to **Row Name** in the **Get Data Table Row** function, but these pins are not compatible. So, you need to first convert the text item to a string by left-clicking and dragging it from the **Item** node and then navigating to **Utilities** | **String** | **To String (Text)**. This will create the first conversion you need:

![Buying items](img/B04548_08_27.jpg)

Lastly, you can just link this converted string to the **Row Name** pin in the **Get Data Table Row** function:

![Buying items](img/B04548_08_28.jpg)

Once done, we have completed the logic for a specific item being selected in the shop. Now, we need to calculate the amount of gold that would be the *value* of each item and subtract it from our total gold. To do this, we must first get the RPG instance of the game so that we can call the game gold. However, since we will need this instance for a number of other variables in this Blueprint, we may want the game instance to be called part of our constructor. Create an **Event Construct** if you have not done so already. Next, link a **Cast To RPGGameInstance** object located at **Casting** under **Utilities**. Then, link the **Get Game Instance** object (located in the **Actions** window under **Game**) to the **Cast To RPGGameInstance** object:

![Buying items](img/B04548_08_29.jpg)

Since we will eventually need to access character parameters, such as HP and MP, when applying our items to the player, we will need to get all the party members, and set a Character Target similar to what we did in previous chapters. To do this, create a new variable:

![Buying items](img/B04548_08_30.jpg)

Then, go to **Details** | **Variable**, call the **Character Target** variable, and change its type to **Game Character**, which will reference our game character within the party:

![Buying items](img/B04548_08_31.jpg)

Then, from the **As RPGGame Instance** pin, drag out a line, and pick the **Get Party Members** variable by navigating to **Variables** | **Game Data**:

![Buying items](img/B04548_08_32.jpg)

To the **Party Members** array, link a **GET** function. You need to link **GET** to the Character Target. So, bring in a **SET** version of the new Character Target variable that you created, and link the **GET** function to the **Character Target** pin in **SET Character Target**. Lastly, have the **Cast To RPGGameInstance** execute **SET Character Target**. When you are finished setting up the reference to the game instance and game characters, your constructor will look like this:

![Buying items](img/B04548_08_33.jpg)

Now that we have set a reference to our current game instance, we can manipulate the gold. The next thing you need to do is navigate to your **Get Data Table Row** function. Here, left-click and drag the **Out Row** pin within the function, which will give you some limited options; one of these options is to create **Break ItemsData**. This will allow you to access all of the data for each item. Once done, you will have a box that shows all of the data that we created in our **Items_Shop** Data Table:

![Buying items](img/B04548_08_34.jpg)

The logic is very simple. Basically, if the user has enough money, allow them to purchase an item and subtract the cost of the item by their game gold. If they do not have enough money, do not let them purchase the item.

To do this, we will create a **Get Game Gold** reference. This can be found by navigating to **Class** | **RPGGame Instance** if **Context Sensitive** is unchecked:

![Buying items](img/B04548_08_35.jpg)

Once it is created, link the reference to **As RPGGame Instance** in the **Cast To RPGGame Instance**. You may also notice a **SET** pin that sets **HP** to **5** in the following screenshot; you may add one or leave it alone. This will just indicate that the player starts with 5 HP; this is being done for testing purposes when we test the player consuming a potion; if you decide to use **Set HP** for testing purposes, remember to remove it when you finish play testing:

![Buying items](img/B04548_08_36.jpg)

Now, we will subtract the game gold from the cost of the item being purchased. So, simply create a math function that subtracts an integer from an integer. This math function can be found by navigating to **Math** | **Integer**:

![Buying items](img/B04548_08_37.jpg)

To do the math correctly, we will need to link the game gold to the top pin of the minus function and the gold from **ItemsData** to the lower pin. This will subtract our game gold from the cost of the item:

![Buying items](img/B04548_08_38.jpg)

Here, we need to check whether the player has enough money to purchase the item. So, we will check whether the final product is less than 0\. If it is, we will not allow the player to make the purchase. To make this check, simply use another math function, named **Integer < Integer**, located at **Integer** under **Math**. You will then compare the final product of the subtraction with 0, as shown here:

![Buying items](img/B04548_08_39.jpg)

Next, create a branch by navigating to **Utilities** | **Flow Control**, and link the condition to the condition of the **Integer < Integer** function you just created. Then, link the **Row Found** pin from the **Get Data Table Row** to execute the branch so that if a row is found, the math can occur:

![Buying items](img/B04548_08_40.jpg)

If the final result is not less than 0, then we need to set the game gold to the subtraction product. To do this, bring in the **SET Game Gold** function by navigating to **Class** | **RPGGame Instance** in the **Actions** window with **Context Sensitive** off:

![Buying items](img/B04548_08_41.jpg)

Link the **Target** pin of **Set Game Gold** to the **As RPGGame Instance** pin from the **Cast to RPGGame Instance** function. Then, link the **Game Gold** pin to the final product of the subtraction operation to get the remaining game gold:

![Buying items](img/B04548_08_42.jpg)

The last thing we need to do is link everything correctly. The remaining link is from the branch; if the less than condition returns false, then it means that we have enough money to buy the product, and we can change the game gold. So, next, link the **False** pin from the branch to execute **SET Game Gold**:

![Buying items](img/B04548_08_43.jpg)

If you were to test this now, you would notice that items can be purchased flawlessly from the shop. However, the problem is that the items are never being populated from the shop to the player's inventory. This is a simple fix. Earlier in this chapter, we already set our inventory screen to be able to get an array that can be stored in the Field Player. We will simply use this array to add the items that we buy to the array, and then, retrieve these items when we open our inventory:

![Buying items](img/B04548_08_44.jpg)

Since we already have a way to gather variables from the Field Player, we will bring in the **Get Array Item** variable by navigating to **Class** | **Field Player**.

We will link the **Target** pin of **Array Item** to the **GET** of the **Get All Actors Of Class** function so that we have full access over the `arrayItem` variable. We will then bring in an **Add** function by navigating to **Utilities** | **Array** in the **Actions** window:

![Buying items](img/B04548_08_45.jpg)

The **Add** function will allow you to add elements to an array while dynamically increasing its size (such as a list). To use this, you will link the array that you want to populate; in this case, **Array Item**. Then, you will need to link the item that you want to add to the array; in this case, **Item**. Lastly, you will need to execute **Add**. We will execute it after the **Gold** value is set. In essence, after the player buys the item, the item will then be added to their inventory:

![Buying items](img/B04548_08_46.jpg)

Your buying mechanics are now complete, and you can now test your shop. You will notice that items can be purchased and these purchased items populate your inventory.

# Using items

Now that you have allowed items to populate the inventory, it is now time to make these items work. At this moment, you should still have a branch at the beginning of your item's **onClicked** button. So far, your branch just goes through a false routine because this routine indicates that the player is interacting with the buttons if they are in the shop. It is now time to create a routine for when the **Inventory Screen** Boolean is true, which means that the player is on the inventory screen.

The initial steps between where we created a **Get Data Table Row** function and set it to the **Item_Shop** Data Table (which takes an item row name and breaks the items into item data) are identical to our previous steps. So, we can simply copy and paste those portions from our previous steps into an empty area in this Blueprint:

![Using items](img/B04548_08_47.jpg)

Next, we will link the **True** pin from our initial branch (that is activated by the **Get All Actors Of Class** function) to execute the **Get Data Table Row** function:

![Using items](img/B04548_08_48.jpg)

We will implement logic that is very similar to the logic that we implemented when purchasing items; but this time, we want to make sure that the user gets the correct amount set to them when using an item. Let's first start with the potion. The potion only uses the HP data. So, what we will need to do is add the HP data from the potion to the character's current HP. To do this, we will first need a Character Target variable. So, bring in a **Get Character Target** function from your **Variable** list:

![Using items](img/B04548_08_49.jpg)

Once you do this, link the **Character Target** variable to **Get HP**:

![Using items](img/B04548_08_50.jpg)

Now that you have access to the current player's HP, you can bring in the **Integer + Integer** function by navigating to **Math** | **Integer**. Simply link the **HP** pin from the **Break ItemsData** node to the top pin in the **Integer + Integer** function, and link the character HP to the bottom pin of the **Integer + Integer** node:

![Using items](img/B04548_08_51.jpg)

Here, we need to check whether the product of the addition is less than the character's maximum HP. If it is, we can use the potion. If it is not, we cannot use the potion. So, let's first bring in the **Get MHP** variable from **Character Target**, which shows what the character's maximum HP is like:

![Using items](img/B04548_08_52.jpg)

Now, we will need to bring in a condition that checks whether an integer is less than another integer. This can be found in the **Actions** window by navigating to **Math** | **Integer**:

![Using items](img/B04548_08_53.jpg)

Next, link the final addition product to the upper pin of the **Integer < Integer** condition, and link **MHP** to the lower pin:

![Using items](img/B04548_08_54.jpg)

We will now make a branch that checks our condition. This branch should be activated only if a row is found (or if the user clicks on an actual item):

![Using items](img/B04548_08_55.jpg)

If the total HP is less than the maximum HP, then this would mean that the condition is true, and we need to remove the item from the inventory using the **Remove from Parent** function located under **Widget**. Then, we need to use the **SET HP** function by navigating to **Class** | **Game Character** and making it equal to the addition of the product item HP and character HP. We will also need to link the **Target** pin of the **SET HP** function to the reference to **Character Target**:

![Using items](img/B04548_08_56.jpg)

If you test this now, the character will be able to use potions, and the potions will be removed on use, but the user won't be able to fully heal because we are only testing to see whether the product of our addition is more than the maximum HP, which only accounts for situations where a potion's healing properties are not fully used. Therefore, the character may never be able to be 100% recovered. To fix this, we will simply create a routine for the **False** branch that will remove the item from the parent, and then, automatically set the HP to the maximum HP. This will solve our problem of not being able to heal our character all the way to their maximum health:

![Using items](img/B04548_08_57.jpg)

When you are done with this, your HP-based items' Blueprint will look like this:

![Using items](img/B04548_08_58.jpg)

If you test this now, you will notice that all of your potions work perfectly in your inventory. The last potion that we did not finish is the ether, but the ether logic is exactly the same as the potion logic, though instead of checking the effects of HP, you are checking the effects of MP. Note that this logic is not specific to any one item, it is dynamic to the point where any item that affects these stats will run using this logic. So, if later on you have a mega potion, you will not have to redo any logic or add new logic, the mega potion is still considered an item and will apply the correct amount of HP that was given to it through the Data Table.

# Summary

At this point, you now have your currency system that interacts with an NPC. You are able to buy items from the NPC and stock as many items as you want in your inventory, and then correctly use them. Using this knowledge, you should easily be able to create more items throughout the game using the same strategies that we covered in the last couple of chapters.

In the next chapter, we will dig deeper into useable items and work with equipping weapons and armor, which will temporarily change the stats of a player.