# Chapter 7. Working with Resource Files

Games have a lot of resources such as images and audio files. Cocos2d-x has a management system of resources. The following topics will be covered in this chapter:

*   Selecting resource files
*   Managing resource files
*   Using SQLite
*   Using .xml files
*   Using .plist files
*   Using .json files

# Selecting resource files

Your game has images of each resolution for multiresolution adaption. If you have resolved to find an image for each resolution, your application logic is very complicated. Cocos2d-x has a search path mechanism for solving this problem. In this recipe, we will explain this search path mechanism.

## Getting ready

If you want to share some resources between different resolutions, then you can put all the shared resources in the `Resources` folder, and put the resolution-specified resources in different folders as shown in the following image.

![Getting ready](img/B00561_07_01.jpg)

`CloseNormal.png` and `CloseSelected.png` are shared resources between different resolutions. However, `HelloWorld.png` is a resolution-specified resource.

## How to do it...

You can set the priority to search resources for Cocos2d-x as follows:

[PRE0]

## How it works...

Cocos2d-x will find `HelloWorld.png` in `Resources/ipad`. Cocos2d-x will use `HelloWorld.png` in this path; that's why it can find this resource in `Resources/ipad`. However, Cocos2d-x cannot find `CloseNormal.png` in `Resources/ipad`. It will find the `Resources` folder that is the next order path. The system can find it in the `Resources` folder and use it.

You should add this code in the `AppDelegate::applicationDidFinishLaunching` method before creating the first scene. Then, the first scene can use this search path setting.

## See also

*   The search path mechanism in the next recipe called *Managing resource files*.

# Managing resource files

Cocos2d-x has an extension that manages resources. It is called `AssetsManagerExtension`. This extension is designed for a hot update of resources such as images and audio files. You can update a new version of resources on your games by using this extension without updating your applications.

## Getting ready

Before using `AssetsManagerExtension`, you should learn about it. This extension has many useful features to help you make the hot update. Some of these features are as follows:

*   Multithread downloading support
*   Two-level progression support—File-level and byte-level progression
*   Compressed ZIP file support
*   Resuming download
*   Detailed progression information and error information
*   Possibility to retry failed assets

You have to prepare a web server, and hence, your application will download resources.

## How to do it...

You need to upload resources and manifest files. In this case, we will update `HelloWorld.png` and a `.zip` file called `test.zip`. This `.zip` file includes some new images. `AssetsManagerExtension` will download resources according to the manifest files. The manifest files are `version.manifest` and `project.manifest`.

The `version.manifest` file contains the following code:

[PRE1]

The `project.manifest` file contains the following code:

[PRE2]

Then, you have to upload these manifest files and new resources.

Next, you have to prepare your application for a hot update. You have to create the `local.manifest` file in your project. The local manifest file should contain the following code:

[PRE3]

You should make a class that manages `AssetsManagerExtension` in your project. Here, we create a class called `ResourceManager`. Firstly, you will create a header file of `ResourceManager`. It is called `ResourceManager.h`. This file contains the following code:

[PRE4]

The next step is to create a `ResourceManager.cpp` file. This file contains the following code:

[PRE5]

Finally, to start updating the resource, use the following code:

[PRE6]

## How it works...

Firstly, we will explain the manifest file and the mechanism of `AssetsManagerExtension`. The manifest files are in the JSON format. Local manifest and version manifest have the following data:

| Keys | Description |
| --- | --- |
| `packageUrl` | The URL where the assets manager will try to request and download all the assets. |
| `remoteVersionUrl` | The remote version manifest file URL that permits one to check the remote version to determine whether a new version has been uploaded to the server. |
| `remoteManifestUrl` | The remote manifest file URL that contains all the asset information. |
| `version` | The version of this manifest file. |

In addition, the remote manifest has the following data in the key called assets.

| Keys | Description |
| --- | --- |
| `key` | Each key represents the relative path of the asset. |
| `Md5` | The `md5` field represents the version information of the asset. |
| `compressed` | When the compressed field is `true`, the downloaded file will be decompressed automatically; this key is optional. |

`AssetsManagerExtension` will execute the hot update in the following steps:

1.  Read the local manifest in the application.
2.  Download the version manifest according to the remote version URL in the local manifest.
3.  Compare the version in the local manifest to the version in the version manifest.
4.  If both versions do not match, `AssetsManagerExtension` downloads the project manifest according to the remote manifest URL in the local manifest.
5.  Compare the `md5` value in the remote manifest to the `md5` of the asset in the application.
6.  If both `md5` values do not match, `AssetsManagerExtension` downloads this asset.
7.  Next time, `AssetsManagerExtension` will use the version manifest that was downloaded instead of the local manifest.

Next, we will explain the `ResourceManager` class. You can execute the hot update as follows:

[PRE7]

You should call the `ResourceManager::updateAssets` method by specifying the path of the local manifest. `ResourceManager::updateAssets` will create an instance of `AssetsManagerEx`, which is the class name of `AssetsManagerExtension`, by specifying the path of the local manifest and the path of the storage in the application.

It will create an instance of `EventListenerAssetsManagerEx` for listening to the progress of the hot update.

If the compressed value is true, `AssetsManagerExtension` will unzip it after downloading it.

You can update assets by calling the `AssetsManagerEx::update` method. During the update, you can get the following events:

| Event | Description |
| --- | --- |
| `ERROR_NO_LOCAL_MANIFEST` | Cannot find the local manifest. |
| `UPDATE_PROGRESSION` | Get the progression of the update. |
| `ERROR_DOWNLOAD_MANIFEST` | Fail to download the manifest file. |
| `ERROR_PARSE_MANIFEST` | Parse error for the manifest file. |
| `ALREADY_UP_TO_DATE` | Already updating assets (The version in the local manifest and the version in the version manifest are equal.). |
| `UPDATE_FINISHED` | Finished updating assets. |
| `UPDATE_FAILED` | Error occurred during updating assets. In this case, the cause of error may be the connection. You should try to update again. |
| `ERROR_UPDATING` | Failed to update. |
| `ERROR_DECOMPRESS` | Error occurred during unzipping. |

`ResourceManager` dispatches the event called `EVENT_PROGRESS` if it catches the event called `UPDATE_PROGRESSION`. If you catch `EVENT_PROGRESS`, you should update the progress label.

Further, it dispatches the event called `EVENT_FINISHED` if it catches the event called `UPDATE_FINISHED`. If you catch `EVENT_FINISHED`, you should refresh all textures. That's why we remove all texture caches and reload the scene.

[PRE8]

# Using SQLite

You can save and load game data easily by using the database in your game. In a smartphone application, the database called SQLite is usually used. SQLite is easy to use. However, you have to set a few things before using it. In this recipe, we will explain how to set up and use SQLite in Cocos2d-x.

## Getting ready

Cocos2d-x doesn't have an SQLite library. You have to add SQLite's source code to Cocos2d-x.

You need to download the source code from the site [http://sqlite.org/download.html](http://sqlite.org/download.html). The latest version at the time of writing this book is version 3.8.10\. You can download this version's `.zip` file and expand it. Then, you can add the resulting files to your project as shown in the following image:

![Getting ready](img/B00561_07_02.jpg)

In this recipe, we will create an original class called `SQLiteManager`. So, you have to add the `SQLiteManager.h` and `SQLiteManager.cpp` files to your project.

Then, if you build for Android, you have to edit `proj.android/jni/Android.mk` as follows:

[PRE9]

## How to do it...

First, you have to edit the `SQLiteManager.h` file as follows:

[PRE10]

Next, you have to edit the `SQLiteManager.cpp` file. This code is a little long. So, we will explain it step by step.

1.  Add the following code for the singleton class:

    [PRE11]

2.  Add the method that opens and closes the database:

    [PRE12]

3.  Add the method that inserts data to the database:

    [PRE13]

4.  Add the method that selects data from the database:

    [PRE14]

5.  Finally, here's how to use this class. To insert data, use the following code:

    [PRE15]

    To select data, use the following code:

    [PRE16]

## How it works...

Firstly, in the constructor method of the `SQLiteManager` class, this class creates a table called data if it does not already exist. The data table is created in SQL as follows:

[PRE17]

In order to use SQLite, you have to include `sqlite3.h` and use the sqlite3 API. This API is in the C language. If you would like to learn it, you should check the website [http://sqlite.org/cintro.html](http://sqlite.org/cintro.html).

We created our database called `test.sqlite` in the sandbox area of the application. If you want to change the location or the name, you should edit the `open` method.

[PRE18]

You can insert data by using the `insert` method to specify the key and the value.

[PRE19]

Further, you can select the value by using the `select` method to specify the key.

[PRE20]

## There's more...

In this recipe, we created the `insert` method and the `select` method. However, you can execute other SQL methods such as `delete` and `replace`. Further, you can make the database match your game. So, you will need to edit this class for your game.

# Using .xml files

XML is often used as an API's return value. Cocos2d-x has the TinyXML2 library that can parse an XML file. In this recipe, we will explain how to parse XML files by using this library.

## Getting ready

Firstly, you need to create an XML file and save it as `test.xml` in the `Resources/res` folder in your project. In this case, we will use the following code:

[PRE21]

To use the TinyXML-2 library, you have to include it and use namespace as follows:

[PRE22]

## How to do it...

You can parse an XML file by using the TinyXML2 library. In the following code, we parse `test.xml` and log each element in it.

[PRE23]

This result is the following log:

[PRE24]

## How it works...

First, you will have to create an instance of `XMLDocument` and then, parse the `.xml` file by using the `XMLDocument::LoadFile` method. To get the root element, you will have to use the `XMLDocument::RootElement` method. Basically, you can get the element by using the `FirstChildElement` method. If it is a continuous element, you can get the next element by using the `NextSiblingElement` method. If there are no more elements, the return value of `NextSiblingElement` will be null.

Finally, you shouldn't forget to delete the instance of `XMLDocment`. That's why you created it using a new operation.

# Using .plist files

PLIST used in OS X and iOS is a property list. The file extension is `.plist`, but in fact, the PLIST format is an XML format. We often use `.plist` files to store game settings and so on. Cocos2d-x has a class through which you can easily use `.plist` files.

## Getting ready

Firstly, you need to create a `.plist` file and save it as `test.plist` to the `Resources/res` folder in your project. In this case, it has two keys, namely `foo` and `bar`. The `foo` key has an integer value of `1`. The `bar` key has a string value of `This is string`. Refer to the following code:

[PRE25]

## How to do it...

You can parse a `.plist` file by using the `FileUtils::getValueMapFromFile` method. In the following code, we parse `test.plist` and log a key value in it.

[PRE26]

## How it works...

You can parse a `.plist` file by specifying the `.plist` file's path to the `FileUtils::getValueMapFromFile` method. After doing so, you get the data from the `.plist` file as a `ValueMap` value. The `ValueMap` class is a wrapper class-based `std::unordered_map`. PLIST's data containers are `Array` and `Dictionary`. After parsing the `.plist` file, `Array` is `std::vector<Value>` and `Dictionary` is `std::unordered_map<std::string, Value>`. Further, you can distinguish the type of value by using the `Value::getType` method. Then, you can get the value by using the `Value::asInt`, `asFloat`, `asDouble`, `asBool`, and `asString` methods.

## There's more...

You can save the `.plist` file from `ValueMap`. By doing so, you can save your game data in the `.plist` file. To save the `.plist` file, use the following code:

[PRE27]

First, you need to set the key value in `ValueMap`. In this case, the values are all of the integer type, but you can set mixed-type values as well. Finally, you need to save the file as a `.plist` file by using the `FileUtils::writeToFile` method.

# Using .json files

We can use the JSON format like the XML format for saving/loading game-related data. JSON is a simpler format than XML. It takes less space to represent the same data than the XML file format. Further, today, it is used as the value of Web API. Cocos2d-x has a JSON parse library called **RapidJSON**. In this recipe, we will explain how to use RapidJSON.

## Getting ready

RapidJSON is usually included in Cocos2d-x. However, you need to include the header files as follows:

[PRE28]

## How to do it...

Firstly, we will parse a JSON string as follows:

[PRE29]

You can parse JSON by using `rapidjson::Document` as follows:

[PRE30]

## How it works...

You can parse JSON by using the `Document::Parse` method and specifying the JSON string. You may get a parse error when you use the `Document::HasParseError` method; you can get a description of this error by using the `Document::GetParseError` method for a string. Further, you can get an element by specifying the hash key and using the `Document::GetString` method.

## There's more...

In a real application, you can get a JSON string from a file. We will now explain how to get this string from a file. First, you need to add a file called `test.json` to the `Resources/res` folder in your project and save it as follows:

[PRE31]

Next, you can get a JSON string from a file as follows:

[PRE32]

You can get the string from the file by using the `FileUtils::getStringFromFile` method. Thereafter, you can parse in the same way. In addition, this JSON string may be an array. You can check whether the format is an array by using the `Document::IsArray` method. Then, you can use a for loop to go through the JSON object in the array.