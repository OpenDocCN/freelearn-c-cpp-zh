# Appendix A. Pop Quiz Answers

# Chapter 3, Qt GUI Programming

## Pop quiz – making signal-slot connections

| Q1 | A slot |
| Q2 | `connect(sender, SIGNAL(toggled(bool)), receiver, SLOT(clear()));` and `connect(sender, &QPushButton::clicked, receiver, &QLineEdit::clear);` |

## Pop quiz – using widgets

| Q1 | `sizeHint` |
| Q2 | `QVariant` |
| Q3 | It represents a functionality that a user can invoke in the program. |

# Chapter 4, Qt Core Essentials

## Pop quiz – Qt core essentials

| Q1 | `QString` |
| Q2 | `((25[0-5]&#124;2[0-4][0-9]&#124;[01]?[0-9][0-9]?)(\.&#124;$)){4}` |
| Q3 | XML |

# Chapter 6, Graphics View

## Pop quiz – mastering Graphics View

| Q1 | You should know, for example, that there is a `QGraphicsSimpleTextItem` that you can use to draw a simple text and that you do not have to deal with `QPainter` yourself in these situations. You should further know that if you have a more complex text containing bold characters you can use `QGraphicsTextItem`, which is able of handling rich text. |
| Q2 | The correct answers these questions pertain to the origin points of the different systems. |
| Q3 | Be aware that `QObject` isn't restricted to the "world of widgets". You can also use it with items. |
| Q4 | The catchword for the correct answer is Parallax Scrolling. |
| Q5 | The correct answer will take into account how you can control the cache and how to affect which parts of the view are actually redrawn when an update is requested. |

# Chapter 7, Networking

## Pop quiz – testing your knowledge

| Q1 | `QNetworkAccessManager`, `QNetworkRequest`, and `QNetworkReply`. |
| Q2 | One has to use `QNetworkRequest::setRawHeader()` with the appropriate HTTP header field "Range". |
| Q3 | `QUrlQuery` |
| Q4 | One has to use `deleteLater()` not delete. |
| Q5 | Both inherit `QAbstractSocket` which inherits `QIODevice`. `QIODevice` is itself also the base class of `QFile`. So the handling-files and sockets have much in common. Thus one does not have to learn a second (complex) API only to communicate with sockets. |
| Q6 | `QUdpSocket` |

# Chapter 8, Scripting

## Pop quiz – scripting

| Q1 | `QScriptEngine::evaluate()` |
| Q2 | `QScriptValue` |
| Q3 | `PyValue` |
| Q4 | They contain all the variables defied within a function invocation so that a set of variables visible from within a script can be modified without affecting the global environment (called sandboxing). |

# Chapter 11, Miscellaneous and Advanced Concepts

## Pop quiz – testing your knowledge

| Q1 | The suffix is `Reading`, for example, `QRotationReading`. |
| Q2 | The class named `QSensorGestureRecognizer`. |
| Q3 | It's the Qt Positioning module and you activate it by adding `QT += positioning` to the project fie. |
| Q4 | One has to overload `QDebug& operator<<()` |
| Q5 | It aborts the execution of the program if `condition` is `false` only if the program was built in the debug mode. |