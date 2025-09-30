# Enabling In-App Purchases with Qt Purchasing

In-app purchasing on mobile phones is essential to generate more income. Qt utilizes system APIs to bring in-app purchases to Qt apps. Android and iOS both have their own app stores, and each store has its own methods for registering products for sale. This is where Qt Purchasing comes in!

In this chapter, we will cover the following topics:

*   Registering in Android and iOS stores
*   Creating an in-app product
*   Restoring purchases

# Registering on Android Google Play

Selling mobile applications is often hit-and-miss; only a few apps that are available to buy actually make money. One of the best ways to make money at the moment is to make your application free to download, but to include in-app purchases. That way, people get to try out your app but also have the opportunity to make purchases if they want enhanced play. This section presumes you have already registered your app to the relevant mobile store. To activate in-app purchases, you first need to register the things you intend to sell. This has its own benefits, as it allows testers to *buy* and install things you intend to sell.

Let's take a look at how to do this on Android devices first:

1.  You will first need to register for a Google Developers account in order to create an application that will be available in Google Play Store, [https://developer.android.com](https://developer.android.com).
2.  You will also need to add and edit the `AndroidManifest.xml` file.

In Qt Creator, navigate to:
Projects | Build | Build Settings | Build Android APK | Create Templates. Here, you will need to edit the Package name, ideally using the convention, `com.<company>.<application name>`. Other naming conventions are of course available, as you can name it anything you want.

3.  The version number must be incremented when you update your package in the Google Play Store. The easiest way to do this is to tick the box labeled Include default permissions for Qt Modules. If not, you need to be sure to add the `uses-permission android:name="com.android.vending.BILLING"` permission.
4.  You will also need to sign this package with your certificates, so create a keystore if you haven't already done this.
5.  In-app purchasing in Google is named **Google Play Billing**, while **Google Play Console** is the name of the website you go through to publish apps to the Google Play Store. You need to register as a developer and pay a registration fee. (For me, it was 25 AUD.) Once the fee is paid, you can set up a merchant account.

6.  After that, it's time to supply information about your application and upload store graphics such as screenshots and promotional videos. This is where you also need to supply contact details for your customers.
7.  You can develop in-app purchasing by making an internal test available only to developers in your organization. Once you have the kinks worked out, and your app enters the alpha stage, you can broaden your test and make a closed test. After that, during beta development, you can have an open test.
8.  On the Google Play Console website, click on your application and navigate to Store presence | In-app products | Managed Products.
9.  Then, click on the blue button labelled **CREATE MANAGED PRODUCT**, as shown in the following screenshot:

![](img/198f86cb-a3ed-407a-9aef-a3186c37c014.png)

That will open a new form titled **New managed product**, as shown in the following screenshot:

![](img/e0331322-e604-4c12-9950-800fa479731e.png)

10.  On this form, complete the following fields:

*   Product ID: This will be used in the Qt app identifier
*   Title
*   Description
*   Status: ACTIVE or INACTIVE
*   Pricing: This is limited to be between $0.99 and $550.00

11.  Then, click Save. You will be registered on Google Play.

If you use the same Product ID for both Android and iOS, it will make the process of developing in-app purchases easier.

# Registering on iOS App Store

You should already be enrolled on the Apple Developer Program and to have accepted all of the necessary agreements relating to tax, banking, and other data.

This section assumes you have already registered an app ID, have signed the relevant agreements, and so on. Registering an in-app purchase on iOS is fairly straightforward:

1.  Navigate to your Apple App Store Connect account and sign in. Click on **Apps**, as we will be registering an application's in-app products.

2.  Click on your app and then select Features. At the top of the page, click on the blue circle that contains a plus sign that is labelled **In-App Purchases (0)**, as shown in the following screenshot:

![](img/8915d530-a536-4bf3-a013-f9364bdeaffd.png)

You can choose from the following options:

| Consumable | Items that are used once by the app and need to be re-purchased |
| Non-Consumable | Items that do not expire but are purchased once |
| Auto-Renewing Subscription | Subscription content that is automatically renewed |
| Non-Renewing Subscription | Subscription content that is not renewed |

3.  You will have to fill in a form for this part of the process, so decide beforehand on the values for the following labelled items:

*   Reference name
*   Product ID
*   Price
*   Tiered prices (start at $1.49)
*   Start Date
*   End Date
*   Display Name
*   Description
*   Screenshot
*   Review Notes

Once you have the product ID, take note of this information for later. You will need it once you create your in-app purchase product with Qt Creator.

# Creating an in-app product

Now, the real fun begins! Suppose you have designed a treasure-hunting game where users move around a map and look for treasure. In this scenario, you may want to offer accelerated gameplay, where users can purchase hints to help them to find the game's hidden treasure.

In our example, we will be selling colors. Colors are really great, as they are collectable and can be sold and traded by users with each other.

When you have developed and registered your app as was mentioned in the last sections on *Registering on iOS App Store* and *Registering on Android Google Play*, you can now develop and test Qt Purchasing. We will start by using QML.

The import line in your QML app to use Qt Purchasing is as follows:

```cpp
import QtPurchasing 1.0
```

Add the following line to the profile:

```cpp
QT += purchasing
```

Now, decide what your in-app purchase is going to be. Note that Qt Purchasing has the following two product types:

*   Consumable
*   Unlockable

Consumable purchases are things such as game tokens that are used once and can be purchased more than once. One example of this is game currency.

Unlockable purchases are features such as additional characters, advertisement removal, and level unlocking that can be re-downloaded, restored, or even transferred.

Our color product is a consumable purchase, enabling users to buy as many colors as they want.

In QtPurchasing, there are the following three QML components:

*   `Product`
*   `Store`
*   `Transaction`

# Store

The `Store` component represents the platform's default marketplace; on Android, it is the Google Play Store, and on iOS, it's the Apple App Store. A `Store` element has one method, `restorePurchases()`, which is used a user wants to restore their purchases.

You can either make `Product` a child of `Store`, or standalone, where the `Store` object is specified by an ID.

# Product

The `Product` component represents an in-app purchase product. The `identifier` property corresponds to the product ID you used in the relevant stores when registering your in-app purchase products.

There are a few things to keep in mind about the `Product` component:

*   `Product` can either be a child of `Store`, or it can be referred to by using the `id` property of `Store` 
*   `Product` can have one of two types: `Product.Consumable` or `Product.Unlockable`
*   A `Product.Consumable` product can be purchased more than once, provided that the purchase has been finalized
*   A `Product.Unlockable` product is purchased once and can be restored

The following code demonstrates a `Product` component that has the type of `Product.Consumable`:

```cpp
Store {
    id: marketPlace
}

Product {
    id: colorProduct
    store:marketPlace
    identifier: "some.store.id"
    type: Product.Consumable
}

Button {
    text: "Buy this color"
    onClicked: {
        colorProduct.purchase()
   }
}
```

Now, it's time to move on to our purchase options. Take at look at the following screenshot:

![](img/2279ba26-a26f-4ae8-8c84-99ec66b8908b.png)

To start the purchase procedure, use the `purchase()` method, which the OK button calls to bring up the following dialog from Google Play Store:

![](img/721ccae4-182d-470a-b472-e595b6359b94.png)

Notice that the payment made in the preceding screenshot is not a real one, but is instead made with the Google test card. No money was exchanged.

You will now want to handle the `onPurchaseSucceeded` and `onPurchaseFailed` signals. If you have products that can be restored, do so in the `onPurchaseRestored` signal, as follows:

```cpp
onPurchaseSucceeded: {
    console.log("sale succeeeded " +  transaction.orderId)
// do something like fill a model view

    transaction.finalize()
}
```

You should also save the transaction. When the app starts up, it queries the Store of any purchases. If the user has purchased products, the `onPurchaseSucceeded` signal will get called with the transition ID for each purchase, so the app knows what purchases have already been made and can act accordingly.

The following screenshot illustrates a successful purchase on Google Play Store:

![](img/91ef0de1-ad7d-4f0a-9609-aab7f3ca4925.png)

If the purchase fails for whatever reason, `onPurchaseFailed` will be called, as follows:

```cpp
onPurchaseFailed: {
    console.log("product purchase failed " + transaction.errorString)
    transaction.finalize()
}
```

You may want to provide a user notification for either of the events we've looked at here, simply to provide clarity and avoid confusion for the user.

# Transaction

`Transaction` represents the purchased product in the market store and contains properties regarding the purchase, including `status`, `orderId`, and a string describing any error that may have occurred. The following table explains these properties:

| `errorString` | A platform-specific string that describes an error |
| `failureReason` | Can be either `NoFailure`, `CanceledByUser`, or `ErrorOccurred` |
| `orderId` | A unique ID issued by the platform store |
| `product` | The `product` object |
| `status` | Can be either `PurchaseApproved`, `PurchaseFailed`, or `PurchaseRestored` |
| `timestamp` | The time a transaction occurred |

`Transaction` has one method: `finalize()`. All transactions need to be finalized whether they succeed or fail.

Once a user has successfully purchased a color, they should see something like the following screenshot:

![](img/c61ef7ce-92ee-45ba-be11-9b64e00b26b3.png)

Note that unlockable products can be restored. Let's move on and take a look at how that can be handled.

# Restoring purchases

A user may want to restore purchases for a number of reasons. Perhaps they have re-installed the app, switched to a new phone, or even reset their existing phone.

Only unlockable products can be restored.

Restoring purchases are initialized via the `restorePurchases()` method, which will then call the `onPurchaseRestored` signal for each purchase that gets restored, as follows:

```cpp
Button {
    text: "Restore Purchases"
    onClicked: {
        colorProduct.restorePurchases()
    }
}
```

In the `product` component, it appears as follows:

```cpp
onPurchaseRestored: {
    // handle product
    console.log("Product restored "+ transaction.orderId)

}
```

As you can see, QML makes it super simple to add in-app purchases and even to restore them if and when the need arises.

# Summary

Qt makes it fairly simple to implement in-app purchases. Most of the work will involve getting your app together, and registering in-app products in your platform's store.

You should now be able to register an in-app purchase product with the relevant app stores. You should also know how to use QML to implement the in-app purchase product and make a store transition. We also explored how to restore any unlockable product purchases.

This chapter was all about mobile phone applications and purchases. In the next chapter, we will look at various cross-compiling methods and at how to debug remotely with an embedded device.