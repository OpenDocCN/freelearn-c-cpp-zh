# 第八章：代码异味和清晰代码实践

本章将涵盖以下主题：

+   代码异味简介

+   清晰代码的概念

+   敏捷和清晰代码实践之间的关系

+   SOLID 设计原则

+   代码重构

+   将代码异味重构为清晰代码

+   将代码异味重构为设计模式

清晰代码是功能上准确并且结构良好编写的源代码。通过彻底的测试，我们可以确保代码在功能上是正确的。我们可以通过代码自审、同行代码审查、代码分析，最重要的是通过代码重构来提高代码质量。

以下是一些清晰代码的特点：

+   易于理解

+   易于增强

+   添加新功能不需要太多的代码更改

+   易于重用

+   不言自明

+   在必要时进行评论

最后，编写清晰代码的最大好处是项目或产品中涉及的开发团队和客户都会感到满意。

# 代码重构

重构有助于提高源代码的结构质量。它不会修改代码的功能，只是改善代码的结构方面的质量。重构使代码更清晰，但有时它可能帮助您改善整体代码性能。但是，您需要明白性能调优与代码重构是不同的。

以下图表展示了开发过程概述：

![](img/18fb6dd3-9b64-448f-b498-ee61513fa728.png)

如何安全地进行代码重构？答案如下：

+   拥抱 DevOps

+   适应测试驱动开发

+   适应行为驱动开发

+   使用验收测试驱动开发

# 代码异味

源代码有两个方面的质量，即**功能**和**结构**。源代码的功能质量可以通过根据客户规格测试代码来实现。大多数开发人员犯的最大错误是他们倾向于在不重构代码的情况下将代码提交到版本控制软件；也就是说，他们一旦认为代码在功能上完成了，就提交代码。

事实上，将代码提交到版本控制通常是一个好习惯，因为这是持续集成和 DevOps 的基础。将代码提交到版本控制后，绝大多数开发人员忽视的是重构代码。重构代码以确保其清晰是非常关键的，没有清晰的代码，敏捷是不可能的。

看起来像面条（意指混乱）的代码需要更多的努力来增强或维护。因此，快速响应客户的请求实际上是不可能的。这就是为什么保持清晰代码对于敏捷至关重要。这适用于您组织中遵循的任何敏捷框架。

# 什么是敏捷？

敏捷就是**快速失败**。敏捷团队能够快速响应客户的需求，而不需要开发团队的任何花哨表演。团队使用的敏捷框架并不是很重要：Scrum、看板、XP 或其他框架。真正重要的是，你是否认真地遵循它们？

作为独立的软件顾问，我个人观察到并学习到一般是谁抱怨敏捷，以及为什么他们抱怨敏捷。

由于 Scrum 是最流行的敏捷框架之一，让我们假设一个产品公司，比如 ABC 科技私人有限公司，决定为他们计划开发的新产品采用 Scrum。好消息是，ABC 科技，就像大多数组织一样，也高效地举办冲刺计划会议、每日站立会议、冲刺回顾、冲刺总结和所有其他 Scrum 仪式。假设 ABC 科技已经确保他们的 Scrum 主管是 Scrum 认证的，产品经理是 Scrum 认证的产品负责人。太好了！到目前为止一切听起来都很好。

假设 ABC Tech 产品团队不使用 TDD、BDD、ATDD 和 DevOps。你认为 ABC Tech 产品团队是敏捷的吗？当然不是。事实上，开发团队将面临繁忙和不切实际的时间表，压力会很大。最终，团队将会非常高的离职率，因为团队不会开心。因此，客户也不会开心，产品的质量会受到严重影响。

你认为 ABC Tech 产品团队出了什么问题？

Scrum 有两套流程，即项目管理流程，由 Scrum 仪式覆盖。然后，还有流程的工程方面，大多数组织并不太关注。这可以从 IT 行业对**Certified SCRUM Developer**（CSD）认证的兴趣或认识中看出。IT 行业对 CSM、CSPO 或 CSP 的兴趣几乎没有对 CSD 的兴趣，而 CSD 对开发人员是必需的。然而，我不认为仅凭认证就能使某人成为专家；它只能显示个人或组织在接受敏捷框架并向客户交付高质量产品方面的严肃性。

除非代码保持清晰，开发团队如何能够快速响应客户的需求？换句话说，除非开发团队的工程师们在产品开发中采用 TDD、BDD、ATDD、持续集成和 DevOps，否则任何团队都无法在 Scrum 或其他敏捷框架中取得成功。

底线是，除非你的组织同等重视工程 Scrum 流程和项目管理 Scrum 流程，否则任何开发团队都不能声称在敏捷中取得成功。

# SOLID 设计原则

SOLID 是一组重要的设计原则的首字母缩写，如果遵循，可以避免代码异味，并在结构和功能上提高代码质量。

如果你的软件架构符合 SOLID 设计原则，那么代码异味可以被预防或重构为清晰的代码。以下原则统称为 SOLID 设计原则：

+   单一职责原则

+   开闭原则

+   里氏替换原则

+   接口隔离

+   依赖反转

最好的部分是，大多数设计模式也遵循并符合 SOLID 设计原则。

让我们在以下各节中逐一讨论前述设计原则。

# 单一职责原则

**单一职责原则**也简称为**SRP**。SRP 表示每个类必须只有一个责任。换句话说，每个类必须只代表一个对象。当一个类代表多个对象时，它往往会违反 SRP 并为多个代码异味打开机会。

例如，让我们以一个简单的`Employee`类为例，如下所示：

![](img/2bcd6820-b96b-4bf2-899a-7556198868eb.png)

在前面的类图中，`Employee`类似乎代表了三个不同的对象：`Employee`、`Address`和`Contact`。因此，它违反了 SRP。根据这个原则，可以从前面的`Employee`类中提取出另外两个类，即`Address`和`Contact`，如下所示：

![](img/769eb1f5-4ade-4e44-8a4f-97a03b47d4f6.png)

为简单起见，本节中使用的类图不显示各自类支持的任何方法，因为我们的重点是通过一个简单的例子理解 SRP。

在前面重构的设计中，Employee 有一个或多个地址（个人和官方）和一个或多个联系人（个人和官方）。最好的部分是，在重构设计后，每个类都只抽象出一件事；也就是说，它只有一个责任。

# 开闭原则

当设计支持添加新功能而无需更改代码或不修改现有源代码时，架构或设计符合**开闭原则**（**OCP**）。正如您所知，根据您的专业行业经验，您遇到的每个项目都以某种方式是可扩展的。这就是您能够向产品添加新功能的方式。然而，当这样的功能扩展是在您不修改现有代码的情况下完成时，设计将符合 OCP。

让我们以一个简单的`Item`类为例，如下所示的代码。为简单起见，`Item`类中只捕获了基本细节：

```cpp
#include <iostream>
#include <string>
using namespace std;
class Item {
       private:
         string name;
         double quantity;
         double pricePerUnit;
       public:
         Item ( string name, double pricePerUnit, double quantity ) {
         this-name = name; 
         this->pricePerUnit = pricePerUnit;
         this->quantity = quantity;
    }
    public double getPrice( ) {
           return quantity * pricePerUnit;
    }
    public String getDescription( ) {
           return name;
    }
};
```

假设前面的`Item`类是一个小商店的简单结算应用程序的一部分。由于`Item`类将能够代表钢笔、计算器、巧克力、笔记本等，它足够通用，可以支持商店处理的任何可计费项目。然而，如果商店老板应该收取**商品和服务税**（**GST**）或**增值税**（**VAT**），现有的`Item`类似乎不支持税收组件。一种常见的方法是修改`Item`类以支持税收组件。然而，如果我们修改现有代码，我们的设计将不符合 OCP。

因此，让我们重构我们的设计，使用访问者设计模式使其符合 OCP。让我们探索重构的可能性，如下所示：

```cpp
#ifndef __VISITABLE_H
#define __VISITABLE_H
#include <string>
 using namespace std;
class Visitor;

class Visitable {
 public:
        virtual void accept ( Visitor * ) = 0;
        virtual double getPrice() = 0;
        virtual string getDescription() = 0;
 };
#endif
```

`Visitable`类是一个带有三个纯虚函数的抽象类。`Item`类将继承`Visitable`抽象类，如下所示：

```cpp
#ifndef __ITEM_H
#define __ITEM_H
#include <iostream>
#include <string>
using namespace std;
#include "Visitable.h"
#include "Visitor.h"
class Item : public Visitable {
 private:
       string name;
       double quantity;
       double unitPrice;
 public:
       Item ( string name, double quantity, double unitPrice );
       string getDescription();
       double getQuantity();
       double getPrice();
       void accept ( Visitor *pVisitor );
 };

 #endif
```

接下来，让我们看一下`Visitor`类，如下所示。它说未来可以实现任意数量的`Visitor`子类来添加新功能，而无需修改`Item`类：

```cpp
class Visitable;
#ifndef __VISITOR_H
#define __VISITOR_H
class Visitor {
 protected:
 double price;

 public:
 virtual void visit ( Visitable * ) = 0;
 virtual double getPrice() = 0;
 };

 #endif
```

`GSTVisitor`类是让我们在不修改`Item`类的情况下添加 GST 功能的类。`GSTVisitor`的实现如下：

```cpp
#include "GSTVisitor.h"

void GSTVisitor::visit ( Visitable *pItem ) {
     price = pItem->getPrice() + (0.18 * pItem->getPrice());
}

double GSTVisitor::getPrice() {
     return price;
}
```

`Makefile`如下所示：

```cpp
all: GSTVisitor.o Item.o main.o
     g++ -o gst.exe GSTVisitor.o Item.o main.o

GSTVisitor.o: GSTVisitor.cpp Visitable.h Visitor.h
     g++ -c GSTVisitor.cpp

Item.o: Item.cpp
     g++ -c Item.cpp

main.o: main.cpp
     g++ -c main.cpp

```

重构后的设计符合 OCP，因为我们将能够在不修改`Item`类的情况下添加新功能。想象一下：如果 GST 计算随时间变化，我们将能够添加`Visitor`的新子类并应对即将到来的变化，而无需修改`Item`类。

# Liskov 替换原则

**Liskov 替换原则**（**LSP**）强调子类遵守基类建立的合同的重要性。在理想的继承层次结构中，随着设计重点向上移动类层次结构，我们应该注意泛化；随着设计重点向下移动类层次结构，我们应该注意专门化。

继承合同是两个类之间的，因此基类有责任制定所有子类都可以遵循的规则，一旦同意，子类同样有责任遵守合同。违背这些设计原则的设计将不符合 LSP。

LSP 说，如果一个方法以基类或接口作为参数，应该能够无条件地替换任何一个子类的实例。

事实上，继承违反了最基本的设计原则：继承是弱内聚和强耦合的。因此，继承的真正好处是多态性，而代码重用与继承相比是微不足道的好处。当 LSP 被违反时，我们无法用其子类实例替换基类实例，最糟糕的是我们无法多态地调用方法。尽管付出使用继承的设计代价，如果我们无法获得多态性的好处，就没有真正使用它的动机。

识别 LSP 违规的技术如下：

+   子类将具有一个或多个带有空实现的重写方法。

+   基类将具有专门的行为，这将迫使某些子类，无论这些专门行为是否符合子类的兴趣

+   并非所有的通用方法都可以多态调用

以下是重构 LSP 违规的方法：

+   将基类中的专门方法移动到需要这些专门行为的子类中。

+   避免强迫模糊相关的类参与继承关系。除非子类是基本类型，否则不要仅仅为了代码重用而使用继承。

+   不要寻找小的好处，比如代码重用，而是在可能的情况下寻找使用多态性或聚合或组合的方法。

# 接口隔离

**接口隔离**设计原则建议为特定目的建模许多小接口，而不是建模代表许多事物的一个更大的接口。在 C++中，具有纯虚函数的抽象类可以被视为一个接口。

让我们举一个简单的例子来理解接口隔离：

```cpp
#include <iostream>
#include <string>
using namespace std;

class IEmployee {
      public:
          virtual string getDoor() = 0;
          virtual string getStreet() = 0;
          virtual string getCity() = 0;
          virtual string getPinCode() = 0;
          virtual string getState() = 0;
          virtual string getCountry() = 0;
          virtual string getName() = 0;
          virtual string getTitle() = 0;
          virtual string getCountryDialCode() = 0;
          virtual string getContactNumber() = 0;
};
```

在上面的例子中，抽象类展示了一个混乱的设计。设计混乱，因为它似乎代表了许多事物，比如员工、地址和联系方式。上述抽象类可以重构的一种方式是将单一接口分解为三个独立的接口：`IEmployee`、`IAddress`和`IContact`。在 C++中，接口只是具有纯虚函数的抽象类：

```cpp
#include <iostream>
#include <string>
#include <list>
using namespace std;

class IEmployee {
  private:
     string firstName, middleName, lastName,
     string title;
     string employeeCode;
     list<IAddress> addresses;
     list<IContact> contactNumbers;
  public:
     virtual string getAddress() = 0;
     virtual string getContactNumber() = 0;
};

class IAddress {
     private:
          string doorNo, street, city, pinCode, state, country;
     public:
          IAddress ( string doorNo, string street, string city, 
            string pinCode, string state, string country );
          virtual string getAddress() = 0;
};

class IContact {
      private:
           string countryCode, mobileNumber;
      public:
           IContact ( string countryCode, string mobileNumber );
           virtual string getMobileNumber() = 0;
};
```

在重构后的代码片段中，每个接口都代表一个对象，因此符合接口隔离设计原则。

# 依赖反转

一个好的设计将是高内聚和低耦合的。因此，我们的设计必须具有较少的依赖性。一个使代码依赖于许多其他对象或模块的设计被认为是一个糟糕的设计。如果**依赖反转**（**DI**）被违反，那么发生在依赖模块中的任何变化都会对我们的模块产生不良影响，导致连锁反应。

让我们举一个简单的例子来理解 DI 的威力。`Mobile`类"拥有"一个`Camera`对象，并且注意到这种拥有的形式是组合。组合是一种独占所有权，其中`Camera`对象的生命周期由`Mobile`对象直接控制：

![](img/b2b7826f-1811-40d8-8bac-b49168c0c40d.png)

正如您在上图中所看到的，`Mobile`类具有`Camera`的实例，而使用的是组合的*has a*形式，这是一种独占所有权关系。

让我们来看一下`Mobile`类的实现，如下所示：

```cpp
#include <iostream>
using namespace std;

class Mobile {
     private:
          Camera camera;
     public:
          Mobile ( );
          bool powerOn();
          bool powerOff();
};

class Camera {
      public:
          bool ON();
          bool OFF();
};

bool Mobile::powerOn() {
       if ( camera.ON() ) {
           cout << "nPositive Logic - assume some complex Mobile power ON logic happens here." << endl;
           return true;
       }
       cout << "nNegative Logic - assume some complex Mobile power OFF logic happens here." << endl;
            << endl;
       return false;
}

bool Mobile::powerOff() {
      if ( camera.OFF() ) {
              cout << "nPositive Logic - assume some complex Mobile power OFF             logic happens here." << endl;
      return true;
 }
      cout << "nNegative Logic - assume some complex Mobile power OFF logic happens here." << endl;
      return false;
}

bool Camera::ON() {
     cout << "nAssume Camera class interacts with Camera hardware heren" << endl;
     cout << "nAssume some Camera ON logic happens here" << endl;
     return true;
}

bool Camera::OFF() {
 cout << "nAssume Camera class interacts with Camera hardware heren" << endl;
 cout << "nAssume some Camera OFF logic happens here" << endl;
 return true;
}
```

在上述代码中，`Mobile`对`Camera`具有实现级别的了解，这是一个糟糕的设计。理想情况下，`Mobile`应该通过接口或具有纯虚函数的抽象类与`Camera`进行交互，因为这样可以将`Camera`的实现与其契约分离。这种方法有助于替换`Camera`而不影响`Mobile`，并且还可以支持一堆`Camera`子类来代替一个单一的相机。

想知道为什么它被称为**依赖注入**（**DI**）或**控制反转**（**IOC**）吗？之所以称之为依赖注入，是因为目前`Camera`的生命周期由`Mobile`对象控制；也就是说，`Camera`由`Mobile`对象实例化和销毁。在这种情况下，如果没有`Camera`，几乎不可能对`Mobile`进行单元测试，因为`Mobile`对`Camera`有硬性依赖。除非实现了`Camera`，否则无法测试`Mobile`的功能，这是一种糟糕的设计方法。当我们反转依赖时，它允许`Mobile`对象使用`Camera`对象，同时放弃控制`Camera`对象的生命周期的责任。这个过程被称为 IOC。优点是你将能够独立单元测试`Mobile`和`Camera`对象，它们由于 IOC 而具有强内聚性和松耦合性。

让我们用 DI 设计原则重构前面的代码：

```cpp
#include <iostream>
using namespace std;

class ICamera {
 public:
 virtual bool ON() = 0;
 virtual bool OFF() = 0;
};

class Mobile {
      private:
 ICamera *pCamera;
      public:
 Mobile ( ICamera *pCamera );
            void setCamera( ICamera *pCamera ); 
            bool powerOn();
            bool powerOff();
};

class Camera : public ICamera {
public:
            bool ON();
            bool OFF();
};

//Constructor Dependency Injection
Mobile::Mobile ( ICamera *pCamera ) {
 this->pCamera = pCamera;
}

//Method Dependency Injection
Mobile::setCamera( ICamera *pCamera ) {
 this->pCamera = pCamera;
}

bool Mobile::powerOn() {
 if ( pCamera->ON() ) {
            cout << "nPositive Logic - assume some complex Mobile power ON logic happens here." << endl;
            return true;
      }
cout << "nNegative Logic - assume some complex Mobile power OFF logic happens here." << endl;
<< endl;
      return false;
}

bool Mobile::powerOff() {
 if ( pCamera->OFF() ) {
           cout << "nPositive Logic - assume some complex Mobile power OFF logic happens here." << endl;
           return true;
}
      cout << "nNegative Logic - assume some complex Mobile power OFF logic happens here." << endl;
      return false;
}

bool Camera::ON() {
       cout << "nAssume Camera class interacts with Camera hardware heren" << endl;
       cout << "nAssume some Camera ON logic happens here" << endl;
       return true;
}

bool Camera::OFF() {
       cout << "nAssume Camera class interacts with Camera hardware heren" << endl;
       cout << "nAssume some Camera OFF logic happens here" << endl;
       return true;
}
```

在前面的代码片段中，变化用粗体标出。IOC 是一种非常强大的技术，它让我们解耦依赖，正如刚才演示的；然而，它的实现非常简单。

# 代码异味

代码异味是一个用来指代缺乏结构质量的代码片段的术语；然而，这段代码可能在功能上是正确的。代码异味违反了 SOLID 设计原则，因此必须认真对待，因为编写不好的代码会导致长期的高昂维护成本。然而，代码异味可以重构为干净的代码。

# 注释异味

作为一名独立的软件顾问，我有很多机会与优秀的开发人员、架构师、质量保证人员、系统管理员、首席技术官和首席执行官、企业家等进行交流和学习。每当我们的讨论涉及到“什么是干净的代码或好的代码？”这个十亿美元的问题时，我几乎在全球范围内得到了一个共同的回答，“好的代码将会有良好的注释。”虽然这部分是正确的，但问题也正是从这里开始。理想情况下，干净的代码应该是不言自明的，不需要注释。然而，有些情况下，注释可以提高整体的可读性和可维护性。并非所有的注释都是代码异味，因此有必要区分好的注释和坏的注释。看一下以下代码片段：

```cpp
if ( condition1 ) {
     // some block of code
}
else if ( condition2 ) {
     // some block of code
}
else {
     // OOPS - the control should not reach here ### Code Smell ###
}
```

我相信你也遇到过这种类型的注释。毋庸置疑，前面的情况是代码异味。理想情况下，开发人员应该重构代码来修复错误，而不是写这样的注释。有一次我在深夜调试一个关键问题，我注意到控制流程到达了一个神秘的空代码块，里面只有一个注释。我相信你也遇到过更有趣的代码，并能想象它带来的挫败感；有时，你也会写这种类型的代码。

一个好的注释将表达代码以特定方式编写的原因，而不是表达代码如何做某事。传达代码如何做某事的注释是代码异味，而传达代码为什么这样做的注释是好的注释，因为代码没有表达为什么部分；因此，好的注释提供了附加值。

# 长方法

当一个方法被确定具有多个责任时，它就变得很长。自然而然，代码超过 20-25 行的方法往往具有多个责任。话虽如此，代码行数更多的方法就更长。这并不意味着代码行数少于 25 行的方法就不长。看一下以下代码片段：

```cpp
void Employee::validateAndSave( ) {
        if ( ( street != "" ) && ( city != "" ) )
              saveEmployeeDetails();
}
```

显然，前面的方法具有多个责任；也就是说，它似乎在验证和保存细节。虽然在保存之前进行验证并没有错，但同一个方法不应该同时做这两件事。因此，前面的方法可以重构为两个具有单一责任的较小方法：

```cpp
private:
void Employee::validateAddress( ) {
     if ( ( street == "" ) || ( city == "" ) )
          throw exception("Invalid Address");
}

public:
void Employee::save() {
      validateAddress();
}
```

在前面的代码中，每个重构后的方法都只负责一个责任。将`validateAddress()`方法作为谓词方法可能很诱人；也就是说，一个返回布尔值的方法。然而，如果`validateAddress()`被写成谓词方法，那么客户端代码将被迫进行`if`检查，这是一个代码异味。通过返回错误代码来处理错误不被认为是面向对象的代码，因此必须使用 C++异常来处理错误。

# 长参数列表

一个面向对象的方法接收较少的参数，因为一个设计良好的对象将具有较强的内聚性和较松散的耦合性。接收太多参数的方法是一个症状，表明做出决定所需的知识是外部接收的，这意味着当前对象本身没有所有的知识来做出决定。

这意味着当前对象的内聚性较弱，耦合性较强，因为它过于依赖外部数据来做决定。成员函数通常倾向于接收较少的参数，因为它们需要的数据成员通常是成员变量。因此，将成员变量传递给成员函数的需求听起来是不自然的。

让我们看看一个方法倾向于接收太多参数的常见原因。最常见的症状和原因在这里列出：

+   对象的内聚性较弱，耦合性较强；也就是说，它过于依赖其他对象

+   这是一个静态方法

+   这是一个放错位置的方法；也就是说，它不属于那个对象

+   这不是面向对象的代码

+   SRP 被违反

以下是重构**长参数列表**（LPL）的方法：

+   避免分散提取和传递数据；考虑传递整个对象，让方法提取所需的细节

+   识别提供参数给接收 LPL 方法的对象，并考虑将方法移动到那里

+   对参数列表进行分组，创建一个参数对象，并将接收 LPL 的方法移到新对象中

# 重复的代码

重复的代码是一个常见的代码异味，不需要太多解释。光是复制和粘贴代码文化本身就不能完全怪罪重复的代码。重复的代码使得代码维护更加繁琐，因为相同的问题可能需要在多个地方修复，而集成新功能需要太多的代码更改，这往往会破坏意外的功能。重复的代码还会增加应用程序的二进制占用空间，因此必须重构为清晰的代码。

# 条件复杂性

条件复杂性代码异味是关于复杂的大条件，随着时间的推移往往变得更大更复杂。这种代码异味可以通过策略设计模式来重构。由于策略设计模式涉及许多相关的对象，因此可以使用`工厂`方法，并且**空对象设计模式**可以用于处理`工厂`方法中不支持的子类：

```cpp
//Before refactoring
void SomeClass::someMethod( ) {
      if (  ! conition1 && condition2 )
         //perform some logic
      else if ( ! condition3 && condition4 && condition5 )
         //perform some logic
      else
         //do something 
} 

//After refactoring
void SomeClass::someMethod() {
     if ( privateMethod1() )
          //perform some logic
     else if ( privateMethod2() )
          //perform some logic
     else
         //do something
}
```

# 大类

一个大类代码异味使得代码难以理解，更难以维护。一个大类可能做了太多事情。大类可以通过将其分解为单一职责的小类来重构。

# 死代码

死代码是被注释掉或者从未被使用或集成的代码。它可以通过代码覆盖工具来检测。通常，开发人员由于缺乏信心而保留这些代码实例，这在遗留代码中更常见。由于每个代码都在版本控制软件工具中被跟踪，死代码可以被删除，如果需要的话，总是可以从版本控制软件中检索回来。

# 原始执念

**原始执念**（PO）是一个错误的设计选择：使用原始数据类型来表示复杂的领域实体。例如，如果使用字符串数据类型来表示日期，虽然起初听起来像一个聪明的主意，但从长远来看，它会带来很多维护麻烦。

假设您使用字符串数据类型表示日期，则以下问题将是一个挑战：

+   您需要根据日期对事物进行排序

+   引入字符串后，日期算术将变得非常复杂

+   根据区域设置支持各种日期格式将变得复杂，使用字符串

理想情况下，日期必须由类表示，而不是原始数据类型。

# 数据类

数据类仅提供 getter 和 setter 函数。虽然它们非常适合将数据从一层传输到另一层，但它们往往会给依赖于数据类的类带来负担。由于数据类不会提供任何有用的功能，与数据类交互或依赖的类最终会使用数据类的数据添加功能。这样，围绕数据类的类违反了 SRP 并且往往会变成一个庞大的类。

# 特征嫉妒

如果某些类对其他类的内部细节了解过多，则被称为特征嫉妒。通常，当其他类是数据类时，就会发生这种情况。代码异味是相互关联的；消除一个代码异味往往会吸引其他代码异味。

# 摘要

在本章中，您学习了以下主题：

+   代码异味和重构代码的重要性

+   SOLID 设计原则：

+   单一责任原则

+   开闭原则

+   里氏替换

+   接口隔离

+   依赖注入

+   各种代码异味：

+   注释异味

+   长方法

+   长参数列表

+   重复代码

+   条件复杂性

+   大类

+   死代码

+   面向对象的代码异味：原始执念

+   数据类

+   特征嫉妒

您还学习了许多重构技术，这将帮助您保持代码更清晰。愉快编码！
