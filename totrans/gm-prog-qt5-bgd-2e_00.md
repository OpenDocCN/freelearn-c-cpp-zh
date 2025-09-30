# Preface

As a leading cross-platform toolkit for all significant desktop, mobile, and embedded platforms, Qt is becoming more popular by the day. This book will help you learn the nitty-gritty of Qt and will equip you with the necessary toolsets to build apps and games. This book is designed as a beginner's guide to take programmers new to Qt from the basics, such as objects, core classes, widgets, and new features in version 5.9, to a level where they can create a custom application with the best practices of programming with Qt.

From a brief introduction of how to create an application and prepare a working environment for both desktop and mobile platforms, we will dive deeper into the basics of creating graphical interfaces and Qt's core concepts of data processing and display. As you progress through the chapters, you'll learn to enrich your games by implementing network connectivity and employing scripting. Delve into Qt Quick, OpenGL, and various other tools to add game logic, design animation, add game physics, handle gamepad input, and build astonishing UIs for games. Toward the end of this book, you'll learn to exploit mobile device features, such as sensors and geolocation services, to build engaging user experiences.

# Who this book is for

This book will be interesting and helpful to programmers and application and UI developers who have basic knowledge of C++. Additionally, some parts of Qt allow you to use JavaScript, so basic knowledge of this language will also be helpful. No previous experience with Qt is required. Developers with up to a year of Qt experience will also benefit from the topics covered in this book.

# To get the most out of this book

You don't need to own or install any particular software before starting to work with the book. A common Windows, Linux, or MacOS system should be sufficient. [Chapter 2](d129202d-f982-4114-b69a-094d0a136fe9.xhtml), *Installation*, contains detailed instructions on how to download and set up everything you'll need.

In this book, you will find several headings that appear frequently:

*   The **Time for action** section contains clear instructions on how to complete a procedure or task.
*   The **What just happened?** section explains the working of the tasks or instructions that you have just completed.
*   The **Have a go hero** sections contain practical challenges that give you ideas to experiment with what you have learned.
*   The **Pop quiz** sections contain short single-choice questions intended to help you test your own understanding. You will find the answers at the end of the book.

While going through the chapters, you will be presented with multiple games and other projects as well as detailed descriptions of how to create them. We advise you to try to create these projects yourself using the instructions we'll give you. If at any point of time you have trouble following the instructions or don't know how to do a certain step, you should take a pick at the example code files to see how it can be done. However, the most important and exciting part of learning is to decide what you want to implement and then find a way to do it, so pay attention to the "Have a go hero" sections or think of your own way to improve each project.

# Download the example code files

You can download the example code files for this book from your account at [www.packtpub.com](http://www.packtpub.com). If you purchased this book elsewhere, you can visit [www.packtpub.com/support](http://www.packtpub.com/support) and register to have the files emailed directly to you.

You can download the code files by following these steps:

1.  Log in or register at [www.packtpub.com](http://www.packtpub.com/support).
2.  Select the SUPPORT tab.
3.  Click on Code Downloads & Errata.
4.  Enter the name of the book in the Search box and follow the onscreen instructions.

Once the file is downloaded, please make sure that you unzip or extract the folder using the latest version of:

*   WinRAR/7-Zip for Windows
*   Zipeg/iZip/UnRarX for Mac
*   7-Zip/PeaZip for Linux

The code bundle for the book is also hosted on GitHub at [https://github.com/PacktPublishing/Game-Programming-Using-Qt-5-Beginners-Guide-Second-Edition](https://github.com/PacktPublishing/Game-Programming-Using-Qt-5-Beginners-Guide-Second-Edition). We also have other code bundles from our rich catalog of books and videos available at **[https://github.com/PacktPublishing/](https://github.com/PacktPublishing/)**. Check them out!

# Conventions used

There are a number of text conventions used throughout this book.

`CodeInText`: Indicates code words in text, database table names, folder names, filenames, file extensions, pathnames, dummy URLs, user input, and Twitter handles. Here is an example: "This API is centered on `QNetworkAccessManager`,which handles the complete communication between your game and the Internet."

A block of code is set as follows:

```cpp
QNetworkRequest request;
request.setUrl(QUrl("http://localhost/version.txt"));
request.setHeader(QNetworkRequest::UserAgentHeader, "MyGame");
m_manager->get(request);
```

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

```cpp
void FileDownload::downloadFinished(QNetworkReply *reply) {
    const QByteArray content = reply->readAll();
    _edit->setPlainText(content);
    reply->deleteLater();
}
```

**Bold**: Indicates a new term, an important word, or words that you see onscreen. For example, words in menus or dialog boxes appear in the text like this. Here is an example: "On the Select Destination Location screen, click on Next to accept the default destination."

Warnings or important notes appear like this.

Tips and tricks appear like this.

# Get in touch

Feedback from our readers is always welcome.

**General feedback**: Email `feedback@packtpub.com` and mention the book title in the subject of your message. If you have questions about any aspect of this book, please email us at `questions@packtpub.com`.

**Errata**: Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you have found a mistake in this book, we would be grateful if you would report this to us. Please visit [www.packtpub.com/submit-errata](http://www.packtpub.com/submit-errata), selecting your book, clicking on the Errata Submission Form link, and entering the details.

**Piracy**: If you come across any illegal copies of our works in any form on the Internet, we would be grateful if you would provide us with the location address or website name. Please contact us at `copyright@packtpub.com` with a link to the material.

**If you are interested in becoming an author**: If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, please visit [authors.packtpub.com](http://authors.packtpub.com/).

# Reviews

Please leave a review. Once you have read and used this book, why not leave a review on the site that you purchased it from? Potential readers can then see and use your unbiased opinion to make purchase decisions, we at Packt can understand what you think about our products, and our authors can see your feedback on their book. Thank you!

For more information about Packt, please visit [packtpub.com](https://www.packtpub.com/).