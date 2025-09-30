# Preface

When I took interest in StyleCop with Version 4.2 in 2008, the tool was heavily criticized on the Internet. These reactions had their roots in the following points:

*   The tool was not open source
*   The rules implemented were arbitrary rules set by Microsoft and were not to the liking of some people
*   They were not related in any way to how the .Net runtime interpreted our code
*   Some tools previously made by Microsoft were in direct contradiction with the rules laid down for StyleCop

If we look today, all the rules of that time continue to be present and StyleCop has finally been widely accepted. Some of this acceptance is certainly due to the fact Microsoft released the sources of StyleCop to the community with Version 4.3.3, but this is not the only reason.

If we look at how we begin development on medium-sized and large projects, one of the first things we do is establish base principles, and included among them is the definition of coding conventions. These rules, stating how our code must look, are here in order to improve readability and maintainability for all developers of the team. The choices made there are fairly arbitrary and depend on the background and preferences of the person (or the development team) who laid them down.

However, after the project begins, it takes a lot of time and code reviews to follow them.

This is where StyleCop becomes handy. Whether the rules laid down comply with the Microsoft set of rule or the team has to make its own from scratch, the tool, once parameterized, can review the code of the project on command or can even be used in continuous integration to enforce the set of rules previously defined.

# What this book covers

*Installing StyleCop with Visual Studio (Simple)* describes the installation process of StyleCop, and teaches how to configure the rules to be executed on a project and how to launch an analysis from Visual Studio.

*Understanding the ReSharper add-in (Simple)* describes the StyleCop addin for ReSharper. We will see its real time analysis and how to easily fix most of the StyleCop violations.

*Automating StyleCop using MSBuild (Simple)* describes how to automate our build process using MSBuild. We will describe which lines need to be added to the MSBuild project in order to enable StyleCop's analysis of it and how to cap the number of violations encountered before the build broke.

*Automating StyleCop using command-line batch (Simple)* describes how to analyze your projects with StyleCop from the command line. For this, we will use a tool named StyleCopCmd, and prepare it to be able to launch the last version of StyleCop.

*Automating StyleCop using NAnt (Intermediate)* describes how to use StyleCopCmd to automate our process using NAnt.

*Integrating StyleCop analysis results in Jenkins/Hudson (Intermediate)* describes how to build a StyleCop analysis job for a project and display its errors.

*Customizing file headers (Simple)* describes how to customize file headers to avoid StyleCop violations, and how we can use Visual Studio templates and snippets to make our life easier while developing.

*Creating custom rules (Intermediate)* describes how to create our own custom rule for the StyleCop engine. We will also see how to add parameters to this rule.

*Integrating StyleCop in your own tool (Advanced)* will show us how to embed StyleCop with your own tools. As an example, we will create a real-time analysis add-in for MonoDevelop/Xamarin Studio.

# What you need for this book

StyleCop is a C# code analyzer; it can be used with Visual Studio or without it.

This book covers the use of StyleCop with and without Visual Studio. In order to follow the different chapters of this book, you will need the following software installed:

*   Visual Studio 2008 professional or superior
*   Jenkins
*   Xamarin Studio or MonoDevelop 4.0

# Who this book is for

This book is intended for .Net developers wishing to discover StyleCop.

# Conventions

In this book, you will find a number of styles of text that distinguish between different kinds of information. Here are some examples of these styles, and an explanation of their meaning.

Code words in text are shown as follows: "it is also possible to include the `Stylecop.targets` file."

A block of code is set as follows:

[PRE0]

When we wish to draw your attention to a particular part of a code block, the relevant lines or items are set in bold:

[PRE1]

Any command-line input or output is written as follows:

[PRE2]

**New terms** and **important words** are shown in bold. Words that you see on the screen, in menus or dialog boxes for example, appear in the text like this: "clicking the **Next** button moves you to the next screen".

### Note

Warnings or important notes appear in a box like this.

### Tip

Tips and tricks appear like this.

# Reader feedback

Feedback from our readers is always welcome. Let us know what you think about this book—what you liked or may have disliked. Reader feedback is important for us to develop titles that you really get the most out of.

To send us general feedback, simply send an e-mail to `<[feedback@packtpub.com](mailto:feedback@packtpub.com)>`, and mention the book title via the subject of your message.

If there is a book that you need and would like to see us publish, please send us a note in the **SUGGEST A TITLE** form on [www.packtpub.com](http://www.packtpub.com) or e-mail `<[suggest@packtpub.com](mailto:suggest@packtpub.com)>`.

If there is a topic that you have expertise in and you are interested in either writing or contributing to a book, see our author guide on [www.packtpub.com/authors](http://www.packtpub.com/authors).

# Customer support

Now that you are the proud owner of a Packt book, we have a number of things to help you to get the most from your purchase.

## Downloading the example code

You can download the example code files for all Packt books you have purchased from your account at [http://www.PacktPub.com](http://www.PacktPub.com). If you purchased this book elsewhere, you can visit [http://www.PacktPub.com/support](http://www.PacktPub.com/support) and register to have the files e-mailed directly to you.

## Errata

Although we have taken every care to ensure the accuracy of our content, mistakes do happen. If you find a mistake in one of our books—maybe a mistake in the text or the code—we would be grateful if you would report this to us. By doing so, you can save other readers from frustration and help us improve subsequent versions of this book. If you find any errata, please report them by visiting [http://www.packtpub.com/support](http://www.packtpub.com/support), selecting your book, clicking on the **errata** **submission** **form** link, and entering the details of your errata. Once your errata are verified, your submission will be accepted and the errata will be uploaded on our website, or added to any list of existing errata, under the Errata section of that title. Any existing errata can be viewed by selecting your title from [http://www.packtpub.com/support](http://www.packtpub.com/support).

## Piracy

Piracy of copyright material on the Internet is an ongoing problem across all media. At Packt, we take the protection of our copyright and licenses very seriously. If you come across any illegal copies of our works, in any form, on the Internet, please provide us with the location address or website name immediately so that we can pursue a remedy.

Please contact us at `<[copyright@packtpub.com](mailto:copyright@packtpub.com)>` with a link to the suspected pirated material.

We appreciate your help in protecting our authors, and our ability to bring you valuable content.

## Questions

You can contact us at `<[questions@packtpub.com](mailto:questions@packtpub.com)>` if you are having a problem with any aspect of the book, and we will do our best to address it.