# Chapter 11. Taking Advantages

The following topics will be covered in this chapter:

*   Using encrypted sprite sheets
*   Using encrypted zip files
*   Using encrypted SQLite files
*   Creating Observer Pattern
*   Networking with HTTP

# Introduction

Until now, we have explained basic technical information in Cocos2d-x. It supports the development of games on a smartphone. Actually, you can create your original games using basic functions of Cocos2d-x. However, if your game is a major hit, cheaters might attempt to crack the code. Therefore, there are cases where encryption is needed to prevent unauthorized access to your game data. Encryption is an important aspect in game development because it helps you to protect your code and prevent people from ruining the overall experience of the game, and it also prevents illegal hacking of game. In this chapter, you will learn how to encrypt your game resources.

# Using encrypted sprite sheets

It is pretty easy for a hacker to extract resource files from the application. This is a huge concern for copyright. Sprite sheets can be encrypted very easily using `TexturePacker`. In this recipe, you will learn how to encrypt your sprites to protect them from hackers and cheaters.

## How to do it...

To encrypt sprite sheets using `TexturePacker`, you need to set it on the left pane of `TexturePacker`. Then, you need to follow the steps written here to successfully encrypt your sprite.

1.  Change the Texture format to `zlib compr. PVR(.pvr.ccz, Ver.2)`
2.  Click on the **ContentProtection** icon, and you will see the additional window in which to set the password.
3.  Type the encryption key in the text input area as shown in the following screenshot. You can type in your favorite key. However, it is difficult to type in 32 hex digits and thus, you can just click on the **Create new key** button. After clicking it, you will find that it automatically inputs the **Encryption key**.![How to do it...](img/B0561_11_01.jpg)
4.  Take a note of this encryption key. This is the key you will need to decrypt the files that are encrypted.
5.  Finally, you can publish the encrypted sprite sheet.

## How it works...

Now, let's have a look on how to use these encrypted sprite sheets.

1.  Add the encrypted sprite sheet to your project as shown in the following image:![How it works...](img/B0561_11_02.jpg)
2.  Include the `ZipUtils` class in `HelloWorld.cpp` to decrypt.

    [PRE0]

3.  Set the encrypting key that is used for encryption by `TexturePacker`.

    [PRE1]

4.  Finally, the sprite is created using the encrypted sprite sheet.

    [PRE2]

## There's more…

The application has a lot of sprite sheets normally. You can use each encryption key per sprite sheet. But this might create some confusion. You need to use the same key in all the sprite sheets in your application. The first time, you need to click on the **Create new key** button to create the encryption key. Then, you need to click on the **Save as global key** button to save the encryption key as the global key. Next time, when you create a new encrypted sprite sheet, you can set this encryption key as a global key by clicking on the **Use global key** button.

Now, we will move on to understanding how to check the encrypted sprite sheets. The encrypted sprite sheet's extension is `.ccz`.

1.  Double-click the encrypted file that has the `.ccz` extension.
2.  Launch Texture Packer and you will see the window where you need to enter the decryption key, as shown in the following screenshot:![There's more…](img/B0561_11_03.jpg)
3.  Enter the decryption key or click on the **Use global key** button. If you have saved the key as the global key, then click on the **OK** button.
4.  If the key is the correct key, you will see the sprite sheet as shown in the preceding screenshot:

# Using encrypted zip files

In a smartphone, the game frequently downloads a `zip` file from the server to update resources. These assets are generally the main targets for hackers. They can decode these assets to manipulate information in a game system. Hence, security for these assets is very important. In this case, `zip` is encrypted to protect against cheaters. In this recipe, you will learn how to unzip an encrypted `zip` file with a password.

## Getting ready

Cocos2d-x has an unzip library. However, encryption/decryption is disabled in this library. That's why we have to enable the crypt option in `unzip.cpp`. This file's path is `cocos2d/external/unzip/unzip.cpp`. You will have to comment out line number 71 of `unzip.cpp` to enable the crypt option.

[PRE3]

When we tried to build in Cocos2d-x version 3.7, an error occurred in `unzip.h` in line 46, as shown in the following code:

[PRE4]

You have to edit the following code to remove this error, as shown:

[PRE5]

## How to do it...

First, include the `unzip.h` file to use the unzip library in `HelloWorld.cpp` as shown in the following code:

[PRE6]

Next, let's try to unzip the encrypted zip file with the password. This can be done by adding the following code in `HelloWorld`.cpp:

[PRE7]

Finally, you can unzip the encrypted zip file to use this method by specifying the password. If the password is `cocos2d-x`, you can unzip with the following code:

[PRE8]

## How it works...

1.  Open the encrypted zip file using the `unzOpen` function, as shown:

    [PRE9]

2.  After opening it with the `unzOpen` function, open it again using the `unzOpenCurrentFilePassword` function, as shown here:

    [PRE10]

3.  After that, you can continue in the same way that is used to unzip an unencrypted zip file.

# Using encrypted SQLite files

We often use SQLite to save the user data or game data. SQLite is a powerful and useful database. However, there is a database file in your game's sand box. Cheaters will get it from your game and they will edit it to cheat. In this recipe, you will learn how to encrypt your SQLite and prevent cheaters from editing it.

## Getting ready

We will use the `wxSqlite` library to encrypt SQLite. This is free software. Firstly, you need to install `wxSqlite` in Cocos2d-x and edit some code and set files in Cocos2d-x.

1.  Download the `wxSqlite3` project's zip file. Visit the following url: [http://sourceforge.net/projects/wxcode/files/Components/wxSQLite3/wxsqlite3-3.1.1.zip/download](http://sourceforge.net/projects/wxcode/files/Components/wxSQLite3/wxsqlite3-3.1.1.zip/download)
2.  Expand the zip file.
3.  Create a new folder called `wxsqlite` under `cocos2d/external`.
4.  Copy `sqlite3/secure/src` after expanding the folder to `cocos2d/external/wxsqlite` as shown in the following screenshot:![Getting ready](img/B0561_11_04.jpg)
5.  Add `sqlite3.h` and `sqlite3secure.c` in `wxsqlite/src` that you added in step 4 to your project, as shown in the following screenshot:![Getting ready](img/B0561_11_05.jpg)
6.  Add `-DSQLITE_HAS_CODEC` to `Other C Flags` in **Build Settings** of Xcode, as shown in the following screenshot:![Getting ready](img/B0561_11_06.jpg)
7.  Create a new file called `Android.mk` in `cocos2d/external/wxsqlite`, as shown in the following code:

    [PRE11]

8.  Edit `Android.mk` in `cocos2d/cocos/storage/local-storage`, as shown in the following code:

    [PRE12]

9.  Edit `LocalStorage.cpp` in `cocos2d/cocos/storage/local-storage`. Comment out line 33 and line 180, as shown in the following code.

    `LocalStorage.cpp` line33:

    [PRE13]

    `LocalStorage.cpp` line180:

    [PRE14]

10.  Edit `Android.mk` in `proj.andorid/jni`, as shown in the following code:

    [PRE15]

After this, SQLite is encrypted and can be used in your project.

## How to do it...

1.  You have to include `sqlite3.h` to use SQLite APIs.

    [PRE16]

2.  Create the encrypted database, as shown in the following code:

    [PRE17]

3.  Select the data from the encrypted database, as shown in the following code:

    [PRE18]

## How it works...

Firstly, you have to create the encrypted database with the `pass` phrase. To create it, follow these three steps:

1.  Open the database normally.
2.  Next, set the pass phrase using the `sqlite3_key` function.

    [PRE19]

3.  Finally, execute sql to create tables.

After this, you will need the encrypted database file in the application. You can get it from the path that was printed by CCLOG.

To select data from there, the same method is used. You can get data from the encrypted database using the same pass phrase after opening the database.

## There's more…

You must be wondering whether this database was really encrypted. So let's check it. Open the database using the command line and executing the command as shown:

[PRE20]

If the database is encrypted, you will not be able to open it and an error message will pop up, as shown:

[PRE21]

# Creating Observer Pattern

Event Dispatcher is a mechanism for responding to events such as touching screen, keyboard events and custom events. You can get an event using Event Dispatcher. In addition, you can create `Observer Pattern` in the design patterns using it. In this recipe, you will learn how to use Event Dispatcher and how to create Observer Pattern in Cocos2d-x.

## Getting ready

Firstly, we will go through the details of Observer Pattern. Observer Pattern is a design pattern. When an event occurs, Observer notifies the event about the subjects that are registered in Observer. It is mainly used to implement distributed event handling. Observer Pattern is also a key part in the MVC architecture.

![Getting ready](img/B0561_11_07.jpg)

## How to do it...

We will create a count up label per second in this recipe. When touching a screen, count up labels are created in this position, and then, count up per second using Observer Pattern.

1.  Create `Count` class that is extended `Label` class as shown in the following code:

    [PRE22]

2.  Next, when touching a screen, this label will be created at the touching position and will call the `HelloWorld::countUp` method per second using a scheduler as the following code in `HelloWorld.cpp`:

    [PRE23]

3.  After building and running this project, when you touch the screen, it will create a count up label at the touching position, and then you will see that the labels are counting up per second at the same time.

## How it works...

1.  Add the custom event called `TimeCount`. If `TimeCount` event occurred, then the `Count::countUp` method is called.

    [PRE24]

2.  Don't forget that you need to remove the custom event from `EventDispatcher` when the instance of the `Count` class is removed. If you forget to do that, then the `zombie` instance will be called from `EventDispatcher` when the event occurs and your game will crash.

    [PRE25]

3.  In `HelloWorld.cpp`, call the `HelloWorld::countUp` method using the scheduler. The `HelloWorld::countUp` method calls the custom event called `TimeOut`.

    [PRE26]

    And then, `EventDispatcher` will notify this event to the listed subjects. In this case, the `Count::countUp` method is called.

    [PRE27]

## There's more…

Using `EventDispatcher`, labels count up at the same time. If you use Scheduler instead of `EventDispatcher`, you will notice something different.

Change the `Count::init` method as shown in the following code:

[PRE28]

In this code, use a scheduler by calling the `Count::countUp` method per second. You can see that the labels are not counting up at the same time in this way. Each label is counting up per second, however not at the same time. Using Observer Pattern, a lot of subjects can be called at the same time.

# Networking with HTTP

In recent smartphone games, we normally use an Internet network to update data, download resources, and so on. There aren't any games developed without networking. In this recipe, you will learn how to use networking to download resources.

## Getting ready

You have to include the header file of `network/HttpClient` to use networking.

[PRE29]

If you run it on Android devices, you need to edit `proj.android/AndroidManifest.xml`.

[PRE30]

## How to do it...

In the following code, we will get the response from [http://google.com/](http://google.com/) and then, print the response data as a log.

[PRE31]

## How it works...

1.  Firstly, create an `HttpRequest` instance. The `HttpRequest` class does not have a `create` method. That's why you use `new` for creating the instance.

    [PRE32]

2.  Specify URL and the request type. In this case, set [http://google.com/](http://google.com/) as a request URL and set GET as a request type.

    [PRE33]

3.  Set callback function to receive the data from the server. You can check its success using the `HttpResponse::isSucceed` method. And then you can get the response data using the `HttpResponse::getResponseData` method.

    [PRE34]

4.  You can request networking by calling the `HttpClient::send` method specifying the instance of the `HttpRequest` class. If you are getting a response via the network, then call the callback function as mentioned in Step3.

    [PRE35]

5.  Finally, you have to release the instance of `HttpRequest`. That's why you created it by using `new`.

    [PRE36]

## There's more…

In this section, you will learn how you can get resources from the network using the `HttpRequest` class. In the following code, get the Google log from the network and display it.

[PRE37]

You can see the following window after building and running this code.

![There's more…](img/B0561_11_08.jpg)

### Tip

You have to save the original data in the sandbox. You can get the path of the sandbox using the `FileUtils::getWritablePath` method.