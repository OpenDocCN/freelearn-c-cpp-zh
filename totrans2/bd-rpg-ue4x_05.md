# Chapter 5. Bridging Character Statistics

Now that we have a basic framework set up for our pause menu, we will now focus on the programming aspect of the pause menu.

In this chapter, you will learn how to link character statistics to the pause menu, as discussed in [Chapter 4](ch04.html "Chapter 4. Pause Menu Framework"), *Pause Menu Framework*. By the end of this chapter, you will be able to link any other game statistics you would like to a UMG menu or submenu. We will cover the following topics in this chapter:

*   Getting character data
*   Getting player instances
*   Displaying stats

# Getting character data

At this point, the pause menu is fully designed and ready for data integration. In [Chapter 3](ch03.html "Chapter 3. Exploration and Combat"), *Exploration and Combat*, we had developed means to display some player parameters, such as the player's name, HP, and MP into CombatUI through binding Text Blocks with the **Game Character** variable in order to access character stat values held within **Character Info**. We will do this in a very similar fashion, as we did in the previous chapter, by first opening the **Pause_Main** widget and clicking on the Text Block that we will update with a value.

In this case, we have already designated locations for all our stat values, so we will start with the HP stat that we named **Editable_Soldier_HP**:

![Getting character data](img/B04548_05_01.jpg)

Navigate to **Content** | **Text**, and click on the drop-down menu of **Bind** next to the dropbox. Click on **Create Binding** under the drop-down menu:

![Getting character data](img/B04548_05_02.jpg)

Once you have completed this process, a new function called `Get_Editable_Soldier_HP_Text_0` will be created, and you will automatically be pulled into the graph of the new function. Like in previous binds, the new function will also automatically have **FunctionEntry** with its labeled return:

![Getting character data](img/B04548_05_03.jpg)

We can now create a new **Game Character** reference variable that we will again name **Character Target**:

![Getting character data](img/B04548_05_04.jpg)

Then, we will drag our **Character Target** variable into the `Get_Editable_Soldier_HP_Text_0` graph and set it to **Get**:

![Getting character data](img/B04548_05_05.jpg)

Next, we will create a new node named **Get HP**, which is located under **Variables** | **Character Info**, and link its **Target** pin to the **Character Target** variable pin:

![Getting character data](img/B04548_05_06.jpg)

Lastly, link the HP stat in the **Get Editable Soldier HP Text 0** node to the **Return Value** pin of the **ReturnNode**. This will automatically create a **To Text (Int)** conversion node, which is responsible for converting any integer into a string. When you are finished, your `Get_Editable_Soldier_HP_Text_0` function should look like this:

![Getting character data](img/B04548_05_07.jpg)

# Getting player instances

If you were to test this now, you would see that a value gets created in our pause menu, but the value is **0**. This is not correct because our character is supposed to start with 100 HP according to the character's current stats:

![Getting player instances](img/B04548_05_08.jpg)

The problem occurs because the **Field Player** that accesses the pause menu never assigns any of our character data to **Character Target**. We can easily set the proper character target in Blueprint, but we won't be able to assign any of the character data without exposing our added party members to Blueprint. So, we must first head into `RPGGameInstance.h` and allow the exposure of our current game data to a **Game Data** category of Blueprint in the `UProperty` parameters:

[PRE0]

Your `RPGGameInstance`.`h` file should now look like this:

[PRE1]

Once you have saved and compiled your code, you should be able to properly call any created and added party members in Blueprint, and so we should have read access via the **Field Player** Blueprint.

Now, you can navigate back to the **Field Player** Blueprint and have it get **RPGGameInstance** by creating the **Get Game Instance** function node located under **Game**:

![Getting player instances](img/B04548_05_09.jpg)

Have the **Return Value** of **Get Game Instance** cast to **RPGGameInstance**, which is located under **Utilities** | **Casting** | **RPGGameInstance**. Now that you've got an instance of the **RPGGameInstance** class, you can have the instance refer to the `TArray` of **Party Members**, which holds all your party members, by navigating to the category that you have created for it in **GameData** under **Variables**:

![Getting player instances](img/B04548_05_11.jpg)

Here, we will need to point to the element of the array that holds our soldier character's stats, which is our first element or `0` index of the array, by linking the **Party Members** array to a **GET** function, which can be found by going to **Utilities** | **Array**:

![Getting player instances](img/B04548_05_12.jpg)

### Note

For additional characters, you will need to link another **GET** function to **Party Members** and have the **GET** function point to the element of the array that will point to any other characters (for instance, if you had a healer that is in index 1, your second **GET** function would simply list its index as 1 instead of 0 to pull from the healer's stats). For now, we are just going to focus on the soldier's stats, but you will want to get stats for every character in your party.

Lastly, once we have finished casting **RPGGameInstance**, we will need to set the **Character Target**, which we created in the pause menu, to our **Party Members**. To do this, right-click on your **Event Graph** to create a new action, but uncheck **Context Sensitive** because we are looking for variables that have been declared in a different class (`Pause_Main`). If you navigate to **Class** | **Pause Main**, you will find **Set Character Target**:

![Getting player instances](img/B04548_05_13.jpg)

Here, simply link **Character Target** to the out pin of your **GET** function:

![Getting player instances](img/B04548_05_14.jpg)

Then, set **Character Target** so that it is triggered after **RPGGameInstance** is cast:

![Getting player instances](img/B04548_05_15.jpg)

# Displaying stats

Now, we will need to pick a good spot to cast **RPGGameInstance**. It would be best to cast **RPGGameInstance** after the pause menu has been created, so link the out pin of the **Set Show MouseCursor** node to the in pin of the **Cast To RPGGameInstance**. Then, link the **Return Value** of the **Create Pause_Main Widget** to the **Target** of **Set Character Target**. When you are finished, your **EventGraph** under **FieldPlayer** will look like this:

![Displaying stats](img/B04548_05_16.jpg)

When you are finished, you will see that the HP of the soldier is displayed correctly as the current HP:

![Displaying stats](img/B04548_05_17.jpg)

You can now add the remaining soldier stats to Text Blocks in **Pause_Main** from the pause menu by binding functions and then have these functions return values, such as the character target's MP and name. When you are finished with your soldier character, your **Pause_Main** should look something like this:

![Displaying stats](img/B04548_05_18.jpg)

### Note

We do not yet have levels or experience, we will cover levels and experience in a later chapter.

If you have any other characters, make sure that you add them as well. As mentioned earlier, if you have additional characters in your party, you will need to go back to your **FieldPlayer** Event Graph and create another **GET** function that will get the indexes of your other party members and assign them to new **Character Targets**.

Let's now head back into the **Pause_Inventory** widget and bind character stats to their corresponding Text Blocks. Just like in **Pause_Main**, select a Text Block that you want to bind; in this case, we will grab the **Text Block** to the right of **HP**:

![Displaying stats](img/B04548_05_19.jpg)

Then, simply create a binding for the Text Block, as you did for other Text Blocks. This will, of course, create a binding for a new function that we will return the HP status of the **Character Target**. The issue is that the **Character Target** that we created in **Pause_Main** is a **Game Character** variable local to **Pause_Main**, so we will have to recreate the **Character Target** variable in **Pause_Inventory**. Luckily, the steps are the same as they were; we just need to add a new variable and name it **Character Target**, and then make its type an object reference to **Game Character**:

![Displaying stats](img/B04548_05_20.jpg)

When you are finished, add the **Character Target** variable as a getter, link the **Character Target** variable to get the HP of your character, and link that value to **Return Value** of your **ReturnNode**, just like you did previously. You should have an Event Graph that looks pretty similar to the following screenshot:

![Displaying stats](img/B04548_05_21.jpg)

If you were to test the inventory screen at this point, you would see that the HP value would be 0, but do not panic, you don't need to do much critical thinking to correct the value now that **FieldPlayer** has a general framework for our characters. If you remember, when we cast **RPGGameInstance** after creating the **Pause_Main** widget in the **FieldPlayer** class, we pulled our added party members from our game instance and set it to **Character Target** in **Pause_Main**. We need to perform steps similar to these, but instead of beginning the retrieval of party members in **FieldPlayer**, we must do it in the class in which we created the **Pause_Inventory**, which was created in **Pause_Main**. So, navigate to the Event Graph of the **Pause_Main** widget:

![Displaying stats](img/B04548_05_22.jpg)

In the preceding screenshot, we see that we are creating both the **Pause_Inventory** and **Pause_Equipment** widgets by clicking on their respective buttons. When the screens are created, we remove the current viewport. This is a perfect spot to create our **RPGGameInstance**. So, as mentioned in the previous steps, create a **Get Game Instance**, which is located under **Game**. Then, set the return value to **Cast to RPGGameInstance** by going to **Utilities** | **Casting**, which will then reference the **Party Members** array located at **Game Data** under **Variables**. Here, you will use the **Get** function by going to **Utilities** | **Array**, and link it to the **Party Members** array, pulling index 0\. This is what you should have done, and so far, the steps are identical to what you did in the **FieldPlayer**:

![Displaying stats](img/B04548_05_23.jpg)

The differences set in when you set the **Character Target**. As mentioned earlier, we will set the **Character Target** variable of our newly created **Character Target** variable to **Pause_Inventory**:

![Displaying stats](img/B04548_05_24.jpg)

Once this is done, link the out pin of **Cast To RPGGameInstance** to the in pin of **Set Character Target**. Also, link **Get** to **Character Target**:

![Displaying stats](img/B04548_05_25.jpg)

Lastly, link the out pin of **Add to Viewport** coming from **Pause_Inventory** to the in pin of **Cast To RPGGameInstance** to trigger the retrieval of the character stats, and link the **Return Value** of the **Create Pause_Inventory Widget** to **Target** of **Set Character Target**:

![Displaying stats](img/B04548_05_26.jpg)

At this point, if you test the inventory screen, you will notice that the HP value is being displayed properly:

![Displaying stats](img/B04548_05_28.jpg)

Now that you know how to create references to party members from **Pause_Main**, you can follow the same steps to set each party member as a character target in **Pause_Inventory**. But first, we need to complete all of the stat value displays in **Pause_Inventory** by creating bindings in each stat's respective Text Block and setting the **Return Value** of each Text Block to the value retrieved from **Character Target**.

Once you are finished with the soldier in your **Pause_Inventory**, you will see something that looks like this:

![Displaying stats](img/B04548_05_29.jpg)

At this point, you can easily navigate back to **Pause_Equipment**, create a new **Character Target** variable, then set a **Party Members** to the **Character Target** variable on displaying **Pause_Equipment** in **Pause_Main**, just like you did in **Pause_Inventory**. The **Inventory** and **Equipment** buttons in the **Pause_Main** Event Graph should look something like this when you are done:

![Displaying stats](img/B04548_05_30.jpg)

In the **Pause_Equipment** widget, we can only bind the **AP**, **DP**, **Lk**, and **Name** Text Blocks, as we will be leaving the weapons for later. If you bind these Text Blocks with the newly created **Character Target** in exactly the same way you bound the **Pause_Inventory** Text Blocks, your **Equipment** screen will look like this on testing:

![Displaying stats](img/B04548_05_31.jpg)

At this point, we have finished binding character stats to our pause menu screens for now.

# Summary

In this chapter, we added the current character stats to the pause menu. Now that we are comfortable with UMG, we will be moving on to communicating with NPCs via dialog boxes, along with adding a shop to the game.