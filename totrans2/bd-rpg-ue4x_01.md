# Chapter 1. Getting Started with RPG Design in Unreal

Role-playing games are highly complex things. Even within the RPG genre, there is a diverse range of games with wildly different mechanics and controls.

Before even a single line of code is written, it's important to figure out what kind of RPG you want to make, how the game is played, whether the game should be turn-based or real-time, and what stats the player will have to concern themselves with.

In this chapter, we will cover the following topics which show how to design an RPG before you begin making it:

*   Tools for game design
*   The design and concept phase
*   Describing the game's features and mechanics
*   Tropes in existing RPGs
*   RPG design overview

# Tools for game design

While you can always type everything in Notepad and keep track of design decisions that way, there are a variety of tools available that can help when working on the design document.

Of particular note is the Google suite of tools. These tools come free with a Google account and have many applications, but in this case, we'll take a look at applying them to game design.

## Google Drive

Google Drive is a cloud-based file storage system like Dropbox. It comes free with a Google account and has up to 15 GB of space. Google Drive makes sharing files with others very easy, as long as they also have a Google account. You can also set up permissions, such as who is allowed to modify data (maybe you only want someone to read but not change your design document).

## Google Docs

Integrated with Google Drive is Google Docs, which is a fully featured online word processing application. It includes many features such as live collaborative editing, comments, and a built-in chat sidebar.

The bulk of your design document can be written in Google Docs and shared with any potential collaborators easily.

## Google Spreadsheets

Just as with Google Docs, Google Spreadsheets is also directly integrated with Google Drive. Google Spreadsheets provides an Excel-style interface that can be used to keep track of data in a handy row/column format. You can also enter equations and formulas into cells and calculate their values.

Spreadsheets might be used, for example, to keep track of a game's combat formulas and test them with a range of input values.

Additionally, you can use spreadsheets to keep track of lists of things. For example, you may have a spreadsheet for weapons in your game, including columns for name, type, damage, element, and so on.

## Pencil and paper

Sometimes, nothing beats the trusty method of actually writing things down. If you have a quick idea popped up in your head, it's probably worth quickly jotting it down. Otherwise, you'll most likely forget what the idea was later (even if you think you won't—trust me, you probably will). It doesn't really matter whether you think the idea is worth writing down or not—you can always give it more thought later.

# The design and concept phase

Just as how a writer works from an outline or mind map, or an artist works from a rough sketch, nearly all games start from some sort of a rough concept or design document.

A design document's purpose is to describe nearly everything about a game. In the case of an RPG, it would describe how the player moves around the game world, how the player interacts with enemies and NPCs, how combat works, and more. The design document becomes the basis upon which all the game code is built.

## Concept

Usually, a game starts with a very rough concept.

For example, let's consider the RPG we'll be making in this book. I might have the idea that this game would be a linear turn-based RPG adventure. It's a very rough concept, but that's OK—while it may not be a terribly original concept, it's enough to begin fleshing out and creating a design document from.

## Design

The design document for the game is based on the previous rough concept. Its purpose is to elaborate on the rough concept and describe how it works. For example, while the rough concept was *linear turn-based RPG adventure*, the design document's job is to take that further and describe how the player moves around the world, how the turn-based combat works, combat stats, game over conditions, how the player advances the plot, and a lot more.

You should be able to give your design document to any person and the document should give them a good idea of what your game will be like and how it works. This, in fact, is one of the big strengths of a design document—it's incredibly useful, for example, as a way of ensuring that everyone on a team is on the same page so to speak.

# Describing the game's features and mechanics

So, assuming you have a very rough concept for the game and are now at the design phase, how do you actually describe how the game works?

There are really no rules for how to do this, but you can divide your theoretical game into the important core bits and think about how each one will work, what the rules are, and so on. The more information and the more specific, the better it is. If something is vague, you'll want to expand on it.

For instance, let's take *combat* in our hypothetical turn-based RPG.

*Combatants take turns selecting actions until one team of combatants is dead.*

What order do combatants fight in? How many teams are there?

*Combatants are divided into two teams: the player team and the enemy team. Combatants are ordered by all players and followed by all enemies. They take turns in order to select actions until one team of combatants is dead (either the enemy team or the player team).*

What sort of actions can combatants select?

*Combatants are divided into two teams: the player team and the enemy team. Combatants are ordered by all players and followed by all enemies. Combatants take turns in order to select actions (either attacking a target, casting an ability, or consuming an item) until one team of combatants is dead (either the enemy team or the player team).*

And so on.

# Tropes in existing role-playing games

Even though RPGs can vary wildly, there are still plenty of common themes they frequently share—features that a player expects out of your game.

## Stats and progression

This one goes without saying. Every RPG—and I do mean *every* RPG—has these basic concepts.

Statistics, or *stats*, are the numbers that govern all combat in the game. While the actual stats can vary, it's common to have stats such as max health, max MP, strength, defense, and more.

As players progress through the game, these stats also improve. Their character becomes better in a variety of ways, reaching maximum potential at (or near) the end of the game. The exact way in which this is handled can vary, but most games implement experience points or XP that are earned in combat; when enough XP has been gained, a character's *level* increases, and with it, their stats increase as well.

## Classes

It's common to have *classes* in an RPG. A class can mean a lot, but generally it governs what a character's capabilities are and how that character will progress.

For instance, a *Soldier* class might define that, as an example, a character is able to wield swords, and mainly focuses on increased attack power and defense power as they level up.

## Special abilities

Very few role-playing games can get away with not having magic spells or special abilities of some sort.

Generally, characters will have some kind of *magic* meter that is consumed every time they use one of their special abilities. Additionally, these abilities cannot be cast if the character does not have enough magic (the term for this varies—it might also be called *mana*, *stamina*, or *power*—really, anything to fit the game scenario).

# RPG design overview

With all that aside, we're going to take a look at the design for the RPG we will be developing over the course of this book, which we'll call *Unreal RPG*.

## Setting

The game is set in an open field. Players will encounter enemies who will drop loot experience, which will increase the player's stats.

## Exploration

While not in combat, players explore the world in an isometric view, similar to games such as Diablo. In this view, players can interact with NPCs and props in the world, and also pause the game to manage their party members, inventory, and equipment.

## Dialogue

When interacting with NPCs and props, dialogue may be triggered. Dialogue in the game is primarily text-based. Dialogue boxes may be either linear, the player simply presses a button to advance to the next dialogue page, or multiple-choice. In the case of multiple-choice, the player is presented with a list of options. Each option will then proceed to a different page of dialogue. For instance, an NPC might ask the player a question and allow the player to respond "Yes" or "No", with different responses to each.

## Shopping

A shop UI can also be triggered from a dialogue. For example, a shopkeeper might ask the player whether they want to buy items. If the player chooses "Yes", a shop UI is displayed.

While in a shop, players can buy items from the NPC.

## Gold

Gold can be attained by defeating monsters in battle. This gold is known as a type of enemy drop.

## The pause screen

While the game is paused, players can do the following:

*   View a list of party members and their statuses (health, magic, level, effects, and so on)
*   View abilities that each party member has learned
*   View the amount of gold currently carried
*   Browse an inventory and use items (such as potions, ethers, and so on) on their party members
*   Manage items equipped to each party member (such as weapons, armor, and so on)

## Party members

The player has a list of *party members*. These are all the characters currently on the player's team. For instance, the player may meet a character in a tower who joins their party to aid in combat. Note that in this book, we will only be creating a single party member, but this will lay the foundations of creating additional party members in your future developments.

## Equipment

Each character in the player's party has the following equipment slots:

*   **Armor**: A character's armor generally increases defense
*   **Weapon**: A character's weapon generally provides a boost to their attack power (as given in the attack formula in the *Combat* section of this chapter)

## Classes

Player characters have different classes. A character's class defines the following elements:

*   The experience curve for leveling up
*   How their stats increase as they level up
*   Which abilities they learn as they level up

The game will feature one player character and class. However, based on this player character, we can easily implement more characters and classes, such as a healer or black mage, into the game.

### Soldier

The Soldier class focuses on increasing attack, max HP, and luck. Additionally, special abilities revolve around dealing with lots of damage to enemies.

Therefore, as the Soldier class levels up, they deal more damage to enemies, withstand more hits, and also deliver more critical blows.

## Combat

While exploring the game world, random encounters may be triggered. Additionally, combat encounters can also be triggered from cut scenes and story events.

When an encounter is triggered, the view transitions away from the game world (the field) to an area specifically for combat (the battle area), an arena of sorts.

Combatants are divided into two teams: the enemy team and the player team (consisting of the player's party members).

Each team is lined up, facing each other from the opposite ends of the battle area.

Combatants take turns, with the player team going first, followed by the enemy team. A single round of combat is divided into two phases: decision and action.

Firstly, all combatants choose their action. They can either attack an enemy target or cast an ability.

After all combatants have decided, each combatant executes their action in turn. Most actions have a specific target. If, by the time the combatant executes their action, this target is not available, the combatant will pick the next available target if possible, or else the action will simply fail and the combatant will do nothing.

This cycle continues until either all enemies or players are dead. If all enemies are dead, the player's party members are awarded with XP, and loot may also be gained from the defeated enemies (usually, a random amount of gold).

However, if all players have died, then it is game over.

## Combat stats

Every combatant has the following stats:

*   **Health points**: A character's **health points** (**HP**) represents how much damage the character can take. When HP reaches zero, the character dies.

    HP can be replenished via items or spells, as long as the character is still alive. However, once a character is dead, HP cannot be replenished—the character must first be revived via a special item or spell.

*   **Max health**: This is the maximum amount of HP a character can have at any given time. Healing items and spells only work up to this limit, never beyond. Max health may increase as the character levels up, and can also be temporarily increased by equipping certain items.
*   **Magic points**: A character's **magic points** (**MP**) represents how much magic power they have. Abilities consume some amount of MP, and if the player does not have enough MP for the ability, then that ability cannot be performed. MP can be replenished via items.

    It should be noted that enemies have effectively infinite MP, as their abilities do not cost them any MP.

*   **Max magic**: This is the maximum amount of MP a character can have at any given time. Replenishing items only work up to this limit, never beyond. Max magic may increase as the character levels up, and can also be temporarily increased by equipping certain items.
*   **Attack power**: A character's attack power represents how much damage they can do when they attack an enemy. Weapons have a separate attack power that is added to regular attacks. The exact formula used to deal with damage is as follows:

    *max (player.ATK – enemy.DEF, 0) + player.weapon.ATK*

    So firstly, enemy defense is subtracted from the player's attack power. If this value is less than zero, it is changed to zero. Then, the weapon's attack power is added to the result.

*   **Defense**: A character's defense reduces the damage they take from an enemy attack.

    The exact formula is as given just previously (defense is subtracted from the enemy's base attack value and then the enemy's weapon attack power is added).

*   **Luck**: A character's luck affects that character's chance of landing a critical hit, which will double the damage dealt to an enemy.

    Luck represents the percent chance of dealing with a critical hit. Luck ranges from 0 to 100, representing the range from 0% to 25%, so the formula is as follows:

    *isCriticalHit = random( 0, 100 ) <= ( player.Luck * 0.25 )*

    So, if the player's luck is 10, given that the random number falls at the number 10 within its range of 0 to 100, then the chance of dealing a critical hit is 2.5%.

    The critical hit multiplier is applied after the damage is calculated, as follows:

    *2 * (max( player.ATK – enemy.DEF, 0 ) + player.weapon.ATK )*

## Combat actions

Actions during combat are divided into three categories: attack and ability.

### Attack

Every character has an attack ability that costs zero MP and, for player characters, is shown as the first option in the action menu during the decision phase of a round.

Generally, an attack takes a single enemy target and deals damage to that enemy. The damage formula is as given previously for the *attack power* stat.

### Ability

Every character, as mentioned earlier, has a set of abilities they know. Excluding attack, abilities cost some amount of MP and have a variety of effects. Abilities can have different types of targets, as follows:

*   A single enemy
*   All enemies
*   A single ally
*   All allies

Abilities can heal targets, revive dead targets, remove some effects, summon temporary allies, temporarily increase a character's stats, and more. However, abilities never restore MP.

Abilities have a set MP cost. This is the amount of MP the character must have in order to perform that ability, and the amount of MP that will be consumed upon casting the ability.

## After combat/victory

Once all enemy combatants have died, the player wins the fight. Upon winning the fight, the player is rewarded with random loot, and experience points are divided between party members.

### Loot

Every enemy defines the loot that is received upon defeating the enemy. This includes how much gold is received to defeat this enemy.

### Experience

Each enemy defines how much experience it is worth. After combat, the experience of every defeated enemy is summed up. Then, this value is evenly divided between all currently living players (any party member who has died does not receive any EXP) and rounded up to the nearest integer (for example, if the total experience is 100 and there are three party members, then *100/3 = 33.3333*, which is rounded up to 34).

## Experience and leveling

As party members earn experience, they will level up.

The amount of experience required to go from one level to the next is given by the following formula:

*f(x) = (xa) + c*

Here, *x* is the current level, *a* is a positive value greater than one (affecting how steeply the curve increases), and *c* is the base offset, which is the amount of experience required to go from level 1 to level 2\. This defines a simple exponential value increase. The *a* and *c* values are defined by the character's class.

To get the total amount of experience required to level up from the current level, the preceding formula is calculated and summed for each level up to the current level. For instance, if we want to know how much total EXP is required to get to level 31 (from level 30), we calculate it in the following way:

*f(1) + f(2) + f(3) + … + f(30)*

When a player levels up, their stats increase and they may also learn a new ability. Stat increases and learned abilities are defined on the character class.

The maximum level of any character in the game is 50.

### Stat increases

For a given character class, for every character stat, the class defines a starting value at level 1 and an ending value at 50\. For example, using standard math library functions, the value of attack for any given level would be a simple linear interpolation between the starting value and the ending value, using the character's level (divided by max level) as the interpolation value (the result would then be rounded up to ensure it is a whole integer number).

So, for example, if a soldier's max HP starts at 100 at level 1 and ends at 1,000 at level 50, then at level 25 the soldier's max HP will be 550.

### Learning abilities

Each character class defines a table of abilities. Each entry in the table references which ability will be learned and at what level that ability is learned. When the character levels up, any abilities in the table that has the given level will be added to that character's known abilities.

Any abilities *learned* at level 1 are automatically added to a character's skill set.

## Game over

If all players have died, either during combat or in the field, then the game ends.

# Choosing the right formula

In the preceding sections, the design of this sample game describes a small range of stats for characters in the game. It also outlines a variety of formulas for calculating damage, leveling up, and so on.

One thing to keep in mind is that these stats, values, and formulas are simply there to show you how to implement the core functionality of an RPG. These are not the be-all-end-all of stats or formulas. In fact, the design has intentionally used a limited set of stats and simple formulas to keep the scope simple.

With this in mind, when you're working on your own game, you will have to decide these things for yourself—what stats your characters use, how combat works, and what formulas the game will use to calculate the outcome of battle. So, how do you come up with all of these things on your own?

Unfortunately, the answer is "it depends". There's no silver bullet to balance your game and keep it fun. What stats you use depends on how your combat works (it makes no sense to have a "hit chance" stat if, for instance, your game takes place from a first-person perspective using guns).

Another thing to keep in mind is that the actual values and formulas you use don't matter. What does matter is that the end result is fun, fair, and balanced. It doesn't matter if it takes one, one hundred, or one million experience points to level up if the end result is still fun and feels fair.

### Tip

**Downloading the example code**

You can download the example code files from your account at [http://www.packtpub.com](http://www.packtpub.com) for all the Packt Publishing books you have purchased. If you purchased this book elsewhere, you can visit [http://www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files e-mailed directly to you.

# Summary

In this chapter, we took a look at what tools are at your disposal to design the RPG of your dreams, how important it is to design your game before you begin developing, how to come up with a rough concept and design, and how to describe your game's mechanics. We've also seen an overview of the game that we will be developing over the course of this book.

In the next chapter, we'll start to dive into Unreal and learn about scripting gameplay elements and working with game data in Unreal Engine.