# 第五章：HTTP 和 SSL/TLS

在本章中，我们将涵盖以下主题：

+   实现 HTTP 客户端应用程序

+   实现 HTTP 服务器应用程序

+   为客户端应用程序添加 SSL/TLS 支持

+   为服务器应用程序添加 SSL/TLS 支持

# 简介

本章涵盖两个主要主题。第一个是 HTTP 协议的实现。第二个是 SSL/TLS 协议的使用。让我们简要地考察每个主题。

**HTTP 协议**是一个运行在 TCP 协议之上的应用层协议。它在互联网上被广泛使用，允许客户端应用程序从服务器请求特定资源，并允许服务器将请求的资源传输回客户端。此外，HTTP 还允许客户端上传数据和向服务器发送命令。

HTTP 协议假设几种通信模型或**方法**，每个方法都针对特定目的而设计。最简单的方法称为`GET`，它假设以下事件流程：

1.  HTTP 客户端应用程序（例如，网页浏览器）生成一个包含有关特定资源（位于服务器上）信息的需求消息，并使用 TCP 作为传输层协议将其发送到 HTTP 服务器应用程序（例如，Web 服务器）。

1.  当 HTTP 服务器应用程序收到客户端的请求后，它会解析该请求，从存储（例如，从文件系统或数据库）中提取请求的资源，并将其作为 HTTP 响应消息的一部分发送回客户端。

请求和响应消息的格式由 HTTP 协议定义。

HTTP 协议定义了其他几种方法，允许客户端应用程序主动发送数据或上传资源到服务器，删除服务器上的资源，并执行其他操作。在本章的食谱中，我们将考虑`GET`方法的实现。因为 HTTP 协议的方法在原则上相似，实现其中一种方法可以为其他方法的实现提供很好的提示。

本章还涉及另一个主题：**SSL 和 TLS 协议**。**安全套接字层**（**SSL**）和**传输层安全性**（**TLS**）协议在 TCP 协议之上运行，旨在实现以下两个主要目标：

+   提供一种使用数字证书验证每个通信参与者的方式

+   保护通过底层 TCP 协议传输的数据安全

SSL 和 TLS 协议在 Web 上广泛使用，尤其是在 Web 服务器中。大多数可能向其发送敏感数据（密码、信用卡号码、个人信息等）的 Web 服务器都支持 SSL/TLS 启用通信。在这种情况下，所谓的 HTTPS（通过 SSL 的 HTTP）协议被用来允许客户端验证服务器（有时服务器可能需要验证客户端，尽管这种情况很少见），并通过加密传输的数据来确保数据安全，即使被截获，对犯罪分子来说这些数据也是无用的。

### 注意

Boost.Asio 不包含 SSL/TLS 协议的实现。相反，它依赖于 OpenSSL 库，Boost.Asio 提供了一套类、函数和数据结构，这些结构简化了 OpenSSL 提供的功能的用法，使应用程序的代码更加统一和面向对象。

在本章中，我们不会考虑 OpenSSL 库或 SSL/TLS 协议的细节。这些主题不在此书的范围之内。相反，我们将简要介绍 Boost.Asio 提供的特定工具，这些工具依赖于 OpenSSL 库，并允许在网络应用程序中实现 SSL/TLS 协议的支持。

这两个配方演示了如何构建客户端和服务器应用程序，它们使用 SSL/TLS 协议来确保其通信的安全性。为了使应用程序的 SSL/TLS 相关方面更加生动和清晰，考虑的应用程序的其他方面都被尽可能地简化。客户端和服务器应用程序都是同步的，并基于本书其他章节中的配方。这使我们能够比较基本的 TCP 客户端或服务器应用程序及其支持 SSL/TLS 的高级版本，并更好地理解向分布式应用程序添加 SSL/TLS 支持需要做什么。

# 实现 HTTP 客户端应用程序

HTTP 客户端是分布式软件的一个重要类别，由许多应用程序表示。网络浏览器是这个类的一个突出代表。它们使用 HTTP 协议从网络服务器请求网页。然而，今天 HTTP 协议不仅用于网络。许多分布式应用程序使用此协议来交换任何类型的自定义数据。在设计分布式应用程序时，选择 HTTP 作为通信协议通常比开发自定义协议要好得多。

在这个配方中，我们将考虑使用 Boost.Asio 实现 HTTP 客户端，该客户端满足以下基本要求：

+   支持 HTTP `GET`请求方法

+   异步执行请求

+   支持请求取消

让我们继续到实现部分。

## 如何做到这一点…

由于我们客户应用程序的一个要求是支持取消尚未完成但已启动的请求，我们需要确保所有目标平台都启用了取消功能。因此，我们首先配置 Boost.Asio 库，以便启用请求取消。有关异步操作取消相关问题的更多详细信息，请参阅第二章中的*取消异步操作*配方，*I/O 操作*：

```cpp
#include <boost/predef.h> // Tools to identify the OS.

// We need this to enable cancelling of I/O operations on
// Windows XP, Windows Server 2003 and earlier.
// Refer to "http://www.boost.org/doc/libs/1_58_0/
// doc/html/boost_asio/reference/basic_stream_socket/
// cancel/overload1.html" for details.
#ifdef BOOST_OS_WINDOWS
#define _WIN32_WINNT 0x0501

#if _WIN32_WINNT <= 0x0502 // Windows Server 2003 or earlier.
#define BOOST_ASIO_DISABLE_IOCP
#define BOOST_ASIO_ENABLE_CANCELIO  
#endif
#endif
```

接下来，我们包含 Boost.Asio 库的头文件，以及我们将需要实现应用程序的一些标准 C++库组件的头文件：

```cpp
#include <boost/asio.hpp>

#include <thread>
#include <mutex>
#include <memory>
#include <iostream>

using namespace boost;
```

现在，在我们能够跳转到实现构成我们客户端应用程序的类和函数之前，我们必须进行一项与错误表示和处理相关的准备工作。

在实现 HTTP 客户端应用程序时，我们需要处理三类错误。第一类是由执行 Boost.Asio 函数和类方法时可能发生的许多错误表示的。例如，如果我们对一个表示尚未打开的套接字的对象的 `write_some()` 方法进行调用，该方法将返回操作系统依赖的错误代码（通过抛出异常或通过方法重载使用的输出参数，具体取决于所使用的方法重载），表示在未打开的套接字上执行了无效操作。

第二个类包括由 HTTP 协议定义的既错误又非错误的状态。例如，服务器作为对客户端特定请求的响应返回的状态码 200，表示客户端的请求已成功完成。另一方面，状态码 500 表示在执行请求操作时，服务器发生了错误，导致请求未能完成。

第三个类包括与 HTTP 协议本身相关的错误。如果服务器发送一条消息作为对客户端请求的正确响应，并且这条消息不是一个正确结构的 HTTP 响应，客户端应用程序应该有方法来用错误代码表示这一事实。

第一类错误的错误代码定义在 Boost.Asio 库的源代码中。第二类的状态码由 HTTP 协议定义。第三类在别处没有定义，我们应该在我们的应用程序中自行定义相应的错误代码。

我们定义一个单一的错误代码，它代表一个相当通用的错误，表明从服务器接收到的消息不是一个正确的 HTTP 响应消息，因此客户端无法解析它。让我们将这个错误代码命名为 `invalid_response`：

```cpp
namespace http_errors {
  enum http_error_codes
  {
    invalid_response = 1
  };
```

然后，我们定义一个表示错误类别的类，它包括上面定义的 `invalid_response` 错误代码。让我们将这个类别命名为 `http_errors_category`：

```cpp
  class http_errors_category
    : public boost::system::error_category
  {
  public:
    const char* name() const BOOST_SYSTEM_NOEXCEPT 
    { return "http_errors"; }

    std::string message(int e) const {
      switch (e) {
      case invalid_response:
        return "Server response cannot be parsed.";
        break;
      default:
        return "Unknown error.";
        break;
      }
    }
  };
```

然后，我们定义这个类的静态对象、返回对象实例的函数以及接受我们自定义类型 `http_error_codes` 的错误代码的 `make_error_code()` 函数的重载：

```cpp
const boost::system::error_category&
get_http_errors_category()
{
    static http_errors_category cat;
    return cat;
  }

  boost::system::error_code
    make_error_code(http_error_codes e)
  {
    return boost::system::error_code(
      static_cast<int>(e), get_http_errors_category());
  }
} // namespace http_errors
```

在我们可以在应用程序中使用新的错误代码之前，我们需要执行的最后一个步骤是让 Boost 库知道 `http_error_codes` 枚举的成员应该被视为错误代码。为此，我们将以下结构定义包含到 `boost::system` 命名空间中：

```cpp
namespace boost {
  namespace system {
    template<>
struct is_error_code_enum
<http_errors::http_error_codes>
{
      BOOST_STATIC_CONSTANT(bool, value = true);
    };
  } // namespace system
} // namespace boost
```

由于我们的 HTTP 客户端应用程序将是异步的，当客户端在发起请求时，将需要提供一个指向回调函数的指针，该函数将在请求完成时被调用。我们需要定义一个表示此类回调函数指针的类型。

当回调函数被调用时，需要传递一些参数，这些参数清楚地指明了以下三件事：

+   哪个请求已经完成

+   响应是什么

+   请求是否成功完成，如果没有，则表示发生的错误的错误代码

注意，稍后我们将定义代表 HTTP 请求和 HTTP 响应的 `HTTPRequest` 和 `HTTPResponse` 类，但现在我们使用前置声明。以下是回调函数指针类型声明的样子：

```cpp
class HTTPClient;
class HTTPRequest;
class HTTPResponse;

typedef void(*Callback) (const HTTPRequest& request,
  const HTTPResponse& response,
  const system::error_code& ec);
```

### HTTPResponse 类

现在，我们可以定义一个类来表示作为对请求的响应发送给客户端的 HTTP 响应消息：

```cpp
class HTTPResponse {
  friend class HTTPRequest;
  HTTPResponse() : 
    m_response_stream(&m_response_buf)
  {}
public:

  unsigned int get_status_code() const {
    return m_status_code;
  }

  const std::string& get_status_message() const {
    return m_status_message;
  }

  const std::map<std::string, std::string>& get_headers() {
    return m_headers;
  }

  const std::istream& get_response() const {
    return m_response_stream;
  }

private:
  asio::streambuf& get_response_buf() {
    return m_response_buf;
  }

  void set_status_code(unsigned int status_code) {
    m_status_code = status_code;
  }

  void set_status_message(const std::string& status_message) {
    m_status_message = status_message;
  }

  void add_header(const std::string& name, 
  const std::string& value) 
  {
    m_headers[name] = value;
  }

private:
  unsigned int m_status_code; // HTTP status code.
  std::string m_status_message; // HTTP status message.

  // Response headers.
  std::map<std::string, std::string> m_headers;
  asio::streambuf m_response_buf;
  std::istream m_response_stream;
};
```

`HTTPResponse` 类相当简单。它的私有数据成员代表 HTTP 响应的各个部分，如响应状态码和状态消息，以及响应头和主体。它的公共接口包含返回相应数据成员值的函数，而私有方法允许设置这些值。

将要定义的表示 HTTP 请求的 `HTTPRequest` 类被声明为 `HTTPResponse` 的朋友。我们将看到 `HTTPRequest` 类的实例如何使用 `HTTPResponse` 类的私有方法在收到响应消息时设置其数据成员的值。

### HTTPRequest 类

接下来，我们定义一个类来表示包含基于用户提供的类信息构建 HTTP 请求消息、将其发送到服务器以及接收和解析 HTTP 响应消息的功能的 HTTP 请求。

这个类是我们应用程序的核心，因为它包含了大部分的功能。

然后，我们将定义一个代表 HTTP 客户端的 `HTTPClient` 类，其职责将限于维护所有 `HTTPRequest` 对象共有的单个 `asio::io_service` 类的实例，并作为 `HTTPRequest` 对象的工厂。因此，我们将 `HTTPClient` 类声明为 `HTTPRequest` 类的朋友，并将 `HTTPRequest` 类的构造函数设为私有：

```cpp
class HTTPRequest {
  friend class HTTPClient;

  static const unsigned int DEFAULT_PORT = 80;

  HTTPRequest(asio::io_service& ios, unsigned int id) :
    m_port(DEFAULT_PORT),
    m_id(id),
    m_callback(nullptr),
    m_sock(ios),
    m_resolver(ios),
    m_was_cancelled(false),
    m_ios(ios)  
{}
```

构造函数接受两个参数：一个指向 `asio::io_service` 类对象的引用和一个名为 `id` 的无符号整数。后者包含请求的唯一标识符，由类的用户分配，允许区分不同的请求对象。

然后，我们定义构成类公共接口的方法：

```cpp
public:
  void set_host(const std::string& host) {
    m_host = host;
  }

  void set_port(unsigned int port) {
    m_port = port;
  }

  void set_uri(const std::string& uri) {
    m_uri = uri;
  }

  void set_callback(Callback callback) {
    m_callback = callback;
  }

  std::string get_host() const {
    return m_host;
  }

  unsigned int get_port() const {
    return m_port;
  }

  const std::string& get_uri() const {
    return m_uri;
  }

  unsigned int get_id() const {
    return m_id;
  }

  void execute() {
    // Ensure that precorditions hold.
    assert(m_port > 0);
    assert(m_host.length() > 0);
    assert(m_uri.length() > 0);
    assert(m_callback != nullptr);

    // Prepare the resolving query.
    asio::ip::tcp::resolver::query resolver_query(m_host,
      std::to_string(m_port), 
      asio::ip::tcp::resolver::query::numeric_service);

    std::unique_lock<std::mutex>
      cancel_lock(m_cancel_mux);

    if (m_was_cancelled) {
      cancel_lock.unlock();
      on_finish(boost::system::error_code(
      asio::error::operation_aborted));
      return;
    }

    // Resolve the host name.
    m_resolver.async_resolve(resolver_query,
      this
    {
      on_host_name_resolved(ec, iterator);
    });
  }

  void cancel() {
    std::unique_lock<std::mutex>
      cancel_lock(m_cancel_mux);

    m_was_cancelled = true;

    m_resolver.cancel();

    if (m_sock.is_open()) {
      m_sock.cancel();
    }  
}
```

公共接口包括允许类用户设置和获取 HTTP 请求参数的方法，例如运行服务器的 DNS 名称、协议端口号和请求资源的 URI。此外，还有一个方法允许设置一个回调函数指针，当请求完成时将被调用。

`execute()` 方法启动请求的执行。此外，`cancel()` 方法允许在请求完成之前取消已启动的请求。我们将在食谱的下一部分考虑这些方法的工作原理。

现在，我们定义一组私有方法，其中包含大部分实现细节。首先，我们定义一个用于异步 DNS 名称解析操作回调的方法：

```cpp
private:
  void on_host_name_resolved(
    const boost::system::error_code& ec,
    asio::ip::tcp::resolver::iterator iterator) 
{
    if (ec != 0) {
      on_finish(ec);
      return;
    }

    std::unique_lock<std::mutex>
      cancel_lock(m_cancel_mux);

    if (m_was_cancelled) {
      cancel_lock.unlock();
      on_finish(boost::system::error_code(
      asio::error::operation_aborted));
      return;
    }

    // Connect to the host.
    asio::async_connect(m_sock,
      iterator,
      this
    {
      on_connection_established(ec, iterator);
    });

  }
```

接下来，我们定义一个用于异步连接操作回调的方法，该操作是在刚刚定义的`on_host_name_resolved()`方法中启动的：

```cpp
  void on_connection_established(
    const boost::system::error_code& ec,
    asio::ip::tcp::resolver::iterator iterator) 
{
    if (ec != 0) {
      on_finish(ec);
      return;
    }

    // Compose the request message.
    m_request_buf += "GET " + m_uri + " HTTP/1.1\r\n";

    // Add mandatory header.
    m_request_buf += "Host: " + m_host + "\r\n";

    m_request_buf += "\r\n";

    std::unique_lock<std::mutex>
      cancel_lock(m_cancel_mux);

    if (m_was_cancelled) {
      cancel_lock.unlock();
      on_finish(boost::system::error_code(
      asio::error::operation_aborted));
      return;
    }

    // Send the request message.
    asio::async_write(m_sock,
      asio::buffer(m_request_buf),
      this
    {
      on_request_sent(ec, bytes_transferred);
    });
  }
```

我们定义的下一个方法——`on_request_sent()`——是一个回调，在将请求消息发送到服务器后被调用：

```cpp
  void on_request_sent(const boost::system::error_code& ec,
    std::size_t bytes_transferred) 
{
    if (ec != 0) {
      on_finish(ec);
      return;
    }

    m_sock.shutdown(asio::ip::tcp::socket::shutdown_send);

    std::unique_lock<std::mutex>
      cancel_lock(m_cancel_mux);

    if (m_was_cancelled) {
      cancel_lock.unlock();
      on_finish(boost::system::error_code(
      asio::error::operation_aborted));
      return;
    }

    // Read the status line.
    asio::async_read_until(m_sock,
      m_response.get_response_buf(),
      "\r\n",
      this
    {
      on_status_line_received(ec, bytes_transferred);
    });
  }
```

然后，我们需要另一个回调方法，当从服务器接收到响应消息的第一部分，即**状态行**时，该方法会被调用：

```cpp
  void on_status_line_received(
    const boost::system::error_code& ec,
    std::size_t bytes_transferred)
  {
    if (ec != 0) {
      on_finish(ec);
      return;
    }

    // Parse the status line.
    std::string http_version;
    std::string str_status_code;
    std::string status_message;

    std::istream response_stream(
    &m_response.get_response_buf());
    response_stream >> http_version;

    if (http_version != "HTTP/1.1"){
      // Response is incorrect.
      on_finish(http_errors::invalid_response);
      return;
    }

    response_stream >> str_status_code;

    // Convert status code to integer.
    unsigned int status_code = 200;

    try {
      status_code = std::stoul(str_status_code);
    }
    catch (std::logic_error&) {
      // Response is incorrect.
      on_finish(http_errors::invalid_response);
      return;
    }

    std::getline(response_stream, status_message, '\r');
    // Remove symbol '\n' from the buffer.
    response_stream.get();

    m_response.set_status_code(status_code);
    m_response.set_status_message(status_message);

    std::unique_lock<std::mutex>
      cancel_lock(m_cancel_mux);

    if (m_was_cancelled) {
      cancel_lock.unlock();
      on_finish(boost::system::error_code(
      asio::error::operation_aborted));
      return;
    }

    // At this point the status line is successfully
    // received and parsed.
    // Now read the response headers.
    asio::async_read_until(m_sock,
      m_response.get_response_buf(),
      "\r\n\r\n",
      this
    {
      on_headers_received(ec,
        bytes_transferred);
    });
  }
```

现在，我们定义一个作为回调的方法，当从服务器接收到响应消息的下一段——**响应头块**时，该方法会被调用。我们将它命名为`on_headers_received()`：

```cpp
  void on_headers_received(const boost::system::error_code& ec,
    std::size_t bytes_transferred) 
{
    if (ec != 0) {
      on_finish(ec);
      return;
    }

    // Parse and store headers.
    std::string header, header_name, header_value;
    std::istream response_stream(
    &m_response.get_response_buf());

    while (true) {
      std::getline(response_stream, header, '\r');

      // Remove \n symbol from the stream.
      response_stream.get();

      if (header == "")
        break;

      size_t separator_pos = header.find(':');
      if (separator_pos != std::string::npos) {
        header_name = header.substr(0,
        separator_pos);

        if (separator_pos < header.length() - 1)
          header_value =
          header.substr(separator_pos + 1);
        else
          header_value = "";

        m_response.add_header(header_name,
        header_value);
      }
    }

    std::unique_lock<std::mutex>
      cancel_lock(m_cancel_mux);

    if (m_was_cancelled) {
      cancel_lock.unlock();
      on_finish(boost::system::error_code(
      asio::error::operation_aborted));
      return;
    }

    // Now we want to read the response body.
    asio::async_read(m_sock,
      m_response.get_response_buf(),
      this
    {
      on_response_body_received(ec,
        bytes_transferred);
    });

    return;
  }
```

此外，我们还需要一个处理响应最后一部分——**响应体**的方法。以下方法用作回调，在从服务器接收到响应体后被调用：

```cpp
void on_response_body_received(
const boost::system::error_code& ec,
    std::size_t bytes_transferred) 
{
    if (ec == asio::error::eof)
      on_finish(boost::system::error_code());
    else
      on_finish(ec);  
}
```

最后，我们定义了`on_finish()`方法，它作为所有从`execute()`方法开始执行路径（包括错误路径）的最终点。当请求完成时（无论是成功还是失败），该方法会被调用，其目的是调用`HTTPRequest`类用户提供的回调，通知它请求已完成：

```cpp
  void on_finish(const boost::system::error_code& ec) 
{
    if (ec != 0) {
      std::cout << "Error occured! Error code = "
        << ec.value()
        << ". Message: " << ec.message();
    }

    m_callback(*this, m_response, ec);

    return;
  }
```

我们需要一些与`HTTPRequest`类的每个实例相关联的数据字段。在这里，我们声明类的对应数据成员：

```cpp
private:
  // Request parameters. 
  std::string m_host;
  unsigned int m_port;
  std::string m_uri;

  // Object unique identifier. 
  unsigned int m_id;

  // Callback to be called when request completes. 
  Callback m_callback;

  // Buffer containing the request line.
  std::string m_request_buf;

  asio::ip::tcp::socket m_sock;  
  asio::ip::tcp::resolver m_resolver;

  HTTPResponse m_response;

  bool m_was_cancelled;
  std::mutex m_cancel_mux;

  asio::io_service& m_ios;
```

需要添加的最后一项是关闭括号，以指定`HTTPRequest`类定义的结束：

```cpp
};
```

### HTTPClient 类

我们应用中需要添加的最后一个类是负责以下三个功能的类：

+   为了建立线程策略

+   在 Boost.Asio 事件循环的线程池中创建和销毁线程，以执行异步操作的完成事件

+   作为`HTTPRequest`对象的工厂

我们将这个类命名为`HTTPClient`：

```cpp
class HTTPClient {
public:
  HTTPClient(){
    m_work.reset(new boost::asio::io_service::work(m_ios));

    m_thread.reset(new std::thread([this](){
      m_ios.run();
    }));
  }

  std::shared_ptr<HTTPRequest>
  create_request(unsigned int id) 
  {
    return std::shared_ptr<HTTPRequest>(
    new HTTPRequest(m_ios, id));
  }

  void close() {
    // Destroy the work object. 
    m_work.reset(NULL);

    // Waiting for the I/O thread to exit.
    m_thread->join();
  }

private:
  asio::io_service m_ios;
  std::unique_ptr<boost::asio::io_service::work> m_work;
  std::unique_ptr<std::thread> m_thread;
};
```

### 回调和 main()入口点函数

到目前为止，我们已经有了包含三个类和几个补充数据类型的基本 HTTP 客户端。现在我们将定义两个不是客户端部分，但演示如何使用 HTTP 协议与服务器通信的函数。第一个函数将用作回调，当请求完成时会被调用。它的签名必须与之前定义的`Callback`函数指针类型相匹配。让我们将我们的回调函数命名为`handler()`：

```cpp
void handler(const HTTPRequest& request,
  const HTTPResponse& response,
  const system::error_code& ec)
{
  if (ec == 0) {
    std::cout << "Request #" << request.get_id()
      << " has completed. Response: "
      << response.get_response().rdbuf();
  }
  else if (ec == asio::error::operation_aborted) {
    std::cout << "Request #" << request.get_id()
      << " has been cancelled by the user." 
      << std::endl;
  }
  else {
    std::cout << "Request #" << request.get_id()
      << " failed! Error code = " << ec.value()
      << ". Error message = " << ec.message() 
    << std::endl;
  }

  return;
}
```

我们需要定义的第二个和最后一个函数是`main()`应用程序入口点函数，它使用 HTTP 客户端向服务器发送 HTTP 请求：

```cpp
int main()
{
  try {
    HTTPClient client;

    std::shared_ptr<HTTPRequest> request_one =
      client.create_request(1);

    request_one->set_host("localhost");
    request_one->set_uri("/index.html");
    request_one->set_port(3333);
    request_one->set_callback(handler);

    request_one->execute();

    std::shared_ptr<HTTPRequest> request_two =
      client.create_request(1);

    request_two->set_host("localhost");
    request_two->set_uri("/example.html");
    request_two->set_port(3333);
    request_two->set_callback(handler);

    request_two->execute();

    request_two->cancel();

    // Do nothing for 15 seconds, letting the
    // request complete.
    std::this_thread::sleep_for(std::chrono::seconds(15));

    // Closing the client and exiting the application.
    client.close();
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
};
```

## 它是如何工作的…

现在，让我们考虑我们的 HTTP 客户端是如何工作的。应用程序由五个组件组成，其中包含 `HTTPClient`、`HTTPRequest` 和 `HTTPResponse` 等三个类，以及 `handler()` 回调函数和 `main()` 应用程序入口点函数等两个函数。让我们分别考虑每个组件是如何工作的。 

### `HTTPClient` 类

一个类的构造函数从创建 `asio::io_service::work` 对象的实例开始，以确保在没有任何挂起的异步操作时，运行事件循环的线程不会退出此循环。然后，通过在 `m_ios` 对象上调用 `run()` 方法，生成一个控制线程并将其添加到池中。这就是 `HTTPClient` 类执行其第一个和第二个部分功能的地方，即建立线程策略并将线程添加到池中。

`HTTPClient` 类的第三个功能——作为表示 HTTP 请求的对象的工厂——在其 `create_request()` 公共方法中执行。此方法在空闲内存中创建 `HTTPRequest` 类的实例，并返回一个指向它的共享指针对象。该方法接受一个整数值作为输入参数，该值代表要分配给新创建的请求对象的唯一标识符。此标识符用于区分不同的请求对象。

类的公共接口中的 `close()` 方法销毁 `asio::io_service::work` 对象，允许线程在所有挂起的操作完成时立即退出事件循环。该方法会阻塞，直到所有线程退出。

### `HTTPRequest` 类

让我们从检查其数据成员及其用途开始，来考虑 `HTTPRequest` 类的行为。`HTTPRequest` 类包含 12 个数据成员，其中包含以下内容：

+   请求参数：

    ```cpp
      std::string m_host;
      unsigned int m_port;
      std::string m_uri;
    ```

+   请求的唯一标识符：

    ```cpp
      unsigned int m_id;
    ```

+   指向用户提供的类回调函数的指针，当请求完成时调用：

    ```cpp
      Callback m_callback;
    ```

+   用于存储 HTTP 请求消息的字符串缓冲区：

    ```cpp
      std::string m_request_buf;
    ```

+   用于与服务器通信的套接字对象：

    ```cpp
      asio::ip::tcp::socket m_sock;
    ```

+   用于解析用户提供的服务器主机 DNS 名称的解析器对象：

    ```cpp
      asio::ip::tcp::resolver m_resolver;
    ```

+   表示从服务器接收到的响应的 `HTTPResponse` 类的实例：

    ```cpp
      HTTPResponse m_response;
    ```

+   一个布尔标志和一个支持请求取消功能的 `mutex` 对象（稍后将解释）：

    ```cpp
      bool m_was_cancelled;
      std::mutex m_cancel_mux;
    ```

+   此外，一个引用到由解析器和套接字对象所需的 `asio::io_service` 类的实例。`asio::io_service` 类的单个实例由 `HTTPClient` 类的对象维护：

    ```cpp
      asio::io_service& m_ios;
    ```

`HTTPRequest` 对象的一个实例代表一个单独的 HTTP `GET` 请求。该类设计得如此，以便发送请求需要执行两个步骤。首先，通过在对象上调用相应的设置方法来设置请求的参数和请求完成时调用的回调函数。然后，作为第二步，调用 `execute()` 方法来启动请求执行。当请求完成时，调用回调函数。

`set_host()`、`set_port()`、`set_uri()` 和 `set_callback()` 设置方法允许设置服务器主机 DNS 名称和端口号、请求资源的 URI 以及在请求完成时调用的回调函数。这些方法中的每一个都接受一个参数，并将其值存储在相应的 `HTTPRequest` 对象的数据成员中。

`get_host()`、`get_port()` 和 `get_uri()` 获取器方法返回由相应的设置方法设置的值。`get_id()` 获取器方法返回请求对象的唯一标识符，该标识符在实例化时传递给对象的构造函数。

`execute()` 方法通过启动一系列异步操作来开始请求的执行。每个异步操作执行请求执行过程的一个步骤。

由于请求对象中的服务器主机用 DNS 名称（而不是用 IP 地址）表示，因此在将请求消息发送到服务器之前，必须解析指定的 DNS 名称并将其转换为 IP 地址。因此，请求执行的第一步是 DNS 名称解析。`execute()` 方法从准备解析查询开始，然后调用解析器对象的 `async_resolve()` 方法，指定 `HTTPRequest` 类的 `on_host_name_resolve()` 私有方法作为操作完成回调。

当服务器主机 DNS 名称解析时，调用 `on_host_name_resolved()` 方法。此方法传递两个参数：第一个是一个错误代码，指定操作的状态，第二个是可以用于遍历解析过程产生的端点列表的迭代器。

`on_host_name_resolved()` 方法通过调用 `asio::async_connect()` 自由函数来启动序列中的下一个异步操作，即套接字连接，传递套接字对象 `m_sock` 和迭代器参数给它，以便将套接字连接到第一个有效的端点。`on_connection_established()` 方法被指定为异步连接操作完成回调。

当异步连接操作完成时，会调用`on_connection_established()`方法。传递给它的第一个参数命名为`ec`，它指定了操作完成的状态。如果其值等于零，则表示套接字已成功连接到端点之一。`on_connection_established()`方法使用存储在`HTTPRequest`对象相应数据成员中的请求参数构建 HTTP `GET`请求消息。然后，调用`asio::async_write()`自由函数以异步方式将构建的 HTTP 请求消息发送到服务器。将类中的私有方法`on_request_sent()`指定为在`asio::async_write()`操作完成时调用的回调。

请求发送后，如果发送成功，客户端应用程序必须让服务器知道完整的请求已发送，并且客户端不会发送任何其他内容，通过关闭套接字的发送部分来实现。然后，客户端必须等待来自服务器的响应消息。这正是`on_request_sent()`方法所做的事情。首先，它调用套接字对象的`shutdown()`方法，指定通过将`asio::ip::tcp::socket::shutdown_send`作为参数传递给方法来关闭发送部分。然后，它调用`asio::async_read_until()`自由函数以接收来自服务器的响应。

因为响应可能非常大，而且我们事先不知道其大小，所以我们不想一次性读取它。我们首先只想读取**HTTP 响应状态行**；然后，分析它后，要么继续读取响应的其余部分（如果我们认为需要的话），要么丢弃它。因此，我们将表示 HTTP 响应状态行结束的`\r\n`符号序列作为分隔符参数传递给`asio::async_read_until()`方法。将`on_status_line_received()`方法指定为操作完成回调。

当接收到状态行时，会调用`on_status_line_received()`方法。此方法对状态行进行解析，从中提取指定 HTTP 协议版本、响应状态码和响应状态消息的值。每个值都会进行分析以确保正确性。我们期望 HTTP 版本为 1.1，否则认为响应不正确，并中断请求执行。状态码应该是一个整数值。如果字符串到整数的转换失败，则认为响应不正确，并中断其进一步处理。如果响应状态行正确，则请求执行继续。提取的状态码和状态消息存储在`m_response`成员对象中，并启动请求执行操作序列中的下一个异步操作。现在，我们想要读取响应头块。

根据 HTTP 协议，响应头块以 `\r\n\r\n` 符号序列结束。因此，为了读取它，我们再次调用 `asio::async_read_until()` 自由函数，指定字符串 `\r\n\r\n` 作为分隔符。将 `on_headers_received()` 方法指定为回调。

当接收到响应头块时，会调用 `on_headers_received()` 方法。在这个方法中，响应头块被解析并分解成单独的名字-值对，并作为响应的一部分存储在 `m_response` 成员对象中。

接收并解析了头部信息后，我们想要读取响应的最后部分——响应体。为此，通过调用 `asio::async_read()` 自由函数启动异步读取操作。将 `on_response_body_received()` 方法指定为回调。

最终，会调用 `on_response_body_received()` 方法，通知我们整个响应消息已经接收完毕。因为 HTTP 服务器可能在发送响应消息的最后部分后立即关闭其套接字的发送部分，所以在客户端，最后的读取操作可能以等于 `asio::error::eof` 的错误代码完成。这不应被视为实际错误，而应视为一个正常事件。因此，如果 `on_response_body_received()` 方法以等于 `asio::error::eof` 的 `ec` 参数被调用，我们将默认构造的 `boost::system::error_code` 类对象传递给 `on_finish()` 方法，以指定请求执行已成功完成。否则，`on_finish()` 方法会以表示原始错误代码的参数被调用。`on_finish()` 方法反过来会调用 `HTTPRequest` 类对象客户端提供的回调。

当回调返回时，认为请求处理已完成。

### `HTTPResponse` 类

`HTTPResponse` 类不提供很多功能。它更像是一个包含表示响应不同部分的数据成员的普通数据结构，定义了获取和设置相应数据成员值的获取器和设置器方法。

所有设置方法都是私有的，只有 `HTTPRequest` 类的对象可以访问它们（回想一下，`HTTPRequest` 类被声明为 `HTTPResponse` 类的朋友）。`HTTPRequest` 类的每个对象都有一个数据成员，它是 `HTTPResponse` 类的一个实例。当 `HTTPRequest` 类的对象接收并解析从 HTTP 服务器接收到的响应时，它会设置其 `HTTPResponse` 类成员对象中的值。

### 回调和 `main()` 入口点函数

这些函数演示了如何使用 `HTTPClient` 和 `HTTPRequest` 类来向 HTTP 服务器发送 `GET` HTTP 请求，然后如何使用 `HTTPResponse` 类来获取响应。

`main()`函数首先创建`HTTPClient`类的一个实例，然后使用它创建两个`HTTPRequest`类的实例，每个实例代表一个单独的`GET` HTTP 请求。这两个请求对象都提供了请求参数，然后执行。然而，在第二个请求执行后，第一个请求通过调用其`cancel()`方法被取消。

`handler()`函数，作为在`main()`函数中创建的请求对象的完成回调，无论请求成功、失败或被取消，每次请求完成时都会被调用。`handler()`函数分析传递给它的错误代码和请求以及响应对象，并将相应的消息输出到标准输出流。

## 参见

+   来自第三章的*实现异步 TCP 客户端*配方提供了关于如何实现异步 TCP 客户端的更多信息。

+   来自第六章的*使用定时器*配方，*其他主题*，展示了如何使用 Boost.Asio 提供的定时器。定时器可以用来实现异步操作超时机制。

# 实现 HTTP 服务器应用程序

现在，市场上有很多 HTTP 服务器应用程序。然而，有时需要实现一个定制的应用程序。这可能是一个小型简单的服务器，支持 HTTP 协议的特定子集，可能带有自定义扩展，或者可能不是一个 HTTP 服务器，而是一个支持类似 HTTP 或基于 HTTP 的通信协议的服务器。

在这个配方中，我们将考虑使用 Boost.Asio 实现基本 HTTP 服务器应用程序。以下是我们的应用程序必须满足的一组要求：

+   它应该支持 HTTP 1.1 协议

+   它应该支持`GET`方法

+   它应该能够并行处理多个请求，也就是说，它应该是一个异步并行服务器

事实上，我们已经在考虑实现部分满足指定要求的服务器应用程序。在第四章中，名为*实现服务器应用程序*的配方展示了如何实现一个异步并行 TCP 服务器，该服务器根据一个虚拟的应用层协议与客户端进行通信。所有的通信功能以及协议细节都被封装在一个名为`Service`的单个类中。在该配方中定义的其他所有类和函数在目的上都是基础设施性的，并且与协议细节隔离。因此，当前的配方将基于第四章中的配方，在这里我们只考虑`Service`类的实现，因为其他组件保持不变。

### 注意

注意，在本配方中，我们不考虑应用程序的安全性。在将服务器公开之前，请确保服务器受到保护，尽管它按照 HTTP 协议正确运行，但由于安全漏洞，它可能被犯罪分子所利用。

现在让我们继续实现 HTTP 服务器应用程序。

## 准备中…

因为本配方中演示的应用程序基于第四章中名为*实现异步 TCP 服务器*的配方中的其他应用程序，所以在继续进行本配方之前，有必要熟悉那个配方。

## 如何做…

我们开始我们的应用程序，通过包含包含我们将要使用的数据类型和函数的声明和定义的头文件：

```cpp
#include <boost/asio.hpp>
#include <boost/filesystem.hpp>

#include <fstream>
#include <atomic>
#include <thread>
#include <iostream>

using namespace boost;
```

接下来，我们开始定义提供 HTTP 协议实现的`Service`类。首先，我们声明一个静态常量表，包含 HTTP 状态码和状态消息。表的定义将在`Service`类定义之后给出：

```cpp
class Service {
  static const std::map<unsigned int, std::string>
http_status_table;
```

类的构造函数接受一个参数——指向连接到客户端的套接字实例的共享指针。以下是构造函数的定义：

```cpp
public:
  Service(std::shared_ptr<boost::asio::ip::tcp::socket> sock) :
    m_sock(sock),
    m_request(4096),
    m_response_status_code(200), // Assume success.
    m_resource_size_bytes(0)
  {};
```

接下来，我们定义一个构成`Service`类公共接口的单个方法。该方法通过将指向连接到套接字的客户端实例的指针传递给`Service`类的构造函数，启动与套接字连接的客户端的异步通信会话：

```cpp
  void start_handling() {
    asio::async_read_until(*m_sock.get(),
      m_request,
      "\r\n",
      this
    {
      on_request_line_received(ec,
        bytes_transferred);
    });
  }
```

然后，我们定义一组私有方法，这些方法执行接收和处理客户端发送的请求，解析和执行请求，并将响应发送回去。首先，我们定义一个处理**HTTP 请求行**的方法：

```cpp
private:
  void on_request_line_received(
    const boost::system::error_code& ec,
    std::size_t bytes_transferred) 
{
    if (ec != 0) {
      std::cout << "Error occured! Error code = "
        << ec.value()
        << ". Message: " << ec.message();

      if (ec == asio::error::not_found) {
        // No delimiter has been found in the
        // request message.

        m_response_status_code = 413;
        send_response();

        return;
      }
      else {
        // In case of any other error –
        // close the socket and clean up.
        on_finish();
        return;
      }
    }

    // Parse the request line.
    std::string request_line;
    std::istream request_stream(&m_request);
    std::getline(request_stream, request_line, '\r');
    // Remove symbol '\n' from the buffer.
    request_stream.get();

    // Parse the request line.
    std::string request_method;
    std::istringstream request_line_stream(request_line);
    request_line_stream >> request_method;

    // We only support GET method.
    if (request_method.compare("GET") != 0) {
      // Unsupported method.
      m_response_status_code = 501;
      send_response();

      return;
    }

    request_line_stream >> m_requested_resource;

    std::string request_http_version;
    request_line_stream >> request_http_version;

    if (request_http_version.compare("HTTP/1.1") != 0) {
      // Unsupported HTTP version or bad request.
      m_response_status_code = 505;
      send_response();

      return;
    }

    // At this point the request line is successfully
    // received and parsed. Now read the request headers.
    asio::async_read_until(*m_sock.get(),
      m_request,
      "\r\n\r\n",
      this
    {
      on_headers_received(ec,
        bytes_transferred);
    });

    return;
  }
```

接下来，我们定义一个旨在处理和存储包含请求头的**请求头块**的方法：

```cpp
  void on_headers_received(const boost::system::error_code& ec,
    std::size_t bytes_transferred)  
  {
    if (ec != 0) {
      std::cout << "Error occured! Error code = "
        << ec.value()
        << ". Message: " << ec.message();

      if (ec == asio::error::not_found) {
        // No delimiter has been fonud in the
        // request message.

        m_response_status_code = 413;
        send_response();
        return;
      }
      else {
        // In case of any other error - close the
        // socket and clean up.
        on_finish();
        return;
      }
    }

    // Parse and store headers.
    std::istream request_stream(&m_request);
    std::string header_name, header_value;

    while (!request_stream.eof()) {
      std::getline(request_stream, header_name, ':');
      if (!request_stream.eof()) {
        std::getline(request_stream, 
        header_value, 
      '\r');

        // Remove symbol \n from the stream.
        request_stream.get();
        m_request_headers[header_name] =
        header_value;
      }
    }

    // Now we have all we need to process the request.
    process_request();
    send_response();

    return;
  }
```

此外，我们还需要一个可以执行满足客户端发送的请求所需操作的方法。我们定义了`process_request()`方法，其目的是从文件系统中读取请求资源的内容并将其存储在缓冲区中，以便将其发送回客户端：

```cpp
  void process_request() {
    // Read file.
    std::string resource_file_path =
    std::string("D:\\http_root") +
    m_requested_resource;

    if (!boost::filesystem::exists(resource_file_path)) {
      // Resource not found.
      m_response_status_code = 404;

      return;
    }

    std::ifstream resource_fstream(
    resource_file_path, 
    std::ifstream::binary);

    if (!resource_fstream.is_open()) {
      // Could not open file. 
      // Something bad has happened.
      m_response_status_code = 500;

      return;
    }

    // Find out file size.
    resource_fstream.seekg(0, std::ifstream::end);
    m_resource_size_bytes =
    static_cast<std::size_t>(
    resource_fstream.tellg());

    m_resource_buffer.reset(
    new char[m_resource_size_bytes]);

    resource_fstream.seekg(std::ifstream::beg);
    resource_fstream.read(m_resource_buffer.get(),
    m_resource_size_bytes);

    m_response_headers += std::string("content-length") +
      ": " +
      std::to_string(m_resource_size_bytes) +
      "\r\n";
  }
```

最后，我们定义一个方法来组合响应消息并将其发送到客户端：

```cpp
  void send_response()  {
    m_sock->shutdown(
    asio::ip::tcp::socket::shutdown_receive);

    auto status_line =
      http_status_table.at(m_response_status_code);

    m_response_status_line = std::string("HTTP/1.1 ") +
      status_line +
      "\r\n";

    m_response_headers += "\r\n";

    std::vector<asio::const_buffer> response_buffers;
    response_buffers.push_back(
    asio::buffer(m_response_status_line));

    if (m_response_headers.length() > 0) {
      response_buffers.push_back(
      asio::buffer(m_response_headers));
    }

    if (m_resource_size_bytes > 0) {
      response_buffers.push_back(
      asio::buffer(m_resource_buffer.get(),
      m_resource_size_bytes));
    }

    // Initiate asynchronous write operation.
    asio::async_write(*m_sock.get(),
      response_buffers,
      this
    {
      on_response_sent(ec,
        bytes_transferred);
    });
  }
```

当响应发送完成时，我们需要关闭套接字，让客户端知道已经发送了完整的响应，并且服务器将不再发送更多数据。为此，我们定义了`on_response_sent()`方法：

```cpp
  void on_response_sent(const boost::system::error_code& ec,
    std::size_t bytes_transferred) 
{
    if (ec != 0) {
      std::cout << "Error occured! Error code = "
        << ec.value()
        << ". Message: " << ec.message();
    }

    m_sock->shutdown(asio::ip::tcp::socket::shutdown_both);

    on_finish();
  }
```

我们需要定义的最后一个方法是执行清理并删除`Service`对象实例的方法，当通信会话完成且对象不再需要时：

```cpp
  // Here we perform the cleanup.
  void on_finish() {
    delete this;
  }
```

当然，在我们的类中我们需要一些数据成员。我们声明以下数据成员：

```cpp
private:
  std::shared_ptr<boost::asio::ip::tcp::socket> m_sock;
  boost::asio::streambuf m_request;
  std::map<std::string, std::string> m_request_headers;
  std::string m_requested_resource;

  std::unique_ptr<char[]> m_resource_buffer;  
  unsigned int m_response_status_code;
  std::size_t m_resource_size_bytes;
  std::string m_response_headers;
  std::string m_response_status_line;
};
```

为了完成表示服务的类的定义，我们需要做的最后一件事是定义之前声明的静态成员`http_status_table`并填充数据——HTTP 状态码和相应的状态消息：

```cpp
const std::map<unsigned int, std::string>
  Service::http_status_table = 
{
  { 200, "200 OK" },
  { 404, "404 Not Found" },
  { 413, "413 Request Entity Too Large" },
  { 500, "500 Server Error" },
  { 501, "501 Not Implemented" },
  { 505, "505 HTTP Version Not Supported" }
};
```

我们的`Service`类现在就准备好了。

## 它是如何工作的…

让我们从考虑`Service`类的数据成员开始，然后转向其功能。`Service`类包含以下非静态数据成员：

+   `std::shared_ptr<boost::asio::ip::tcp::socket> m_sock`：这是一个指向连接到客户端的 TCP 套接字对象的共享指针

+   `boost::asio::streambuf m_request`：这是一个缓冲区，请求消息被读取到其中

+   `std::map<std::string, std::string> m_request_headers`：这是一个映射，当解析 HTTP 请求头块时，请求头会被放入其中

+   `std::string m_requested_resource`：这是客户端请求的资源 URI

+   `std::unique_ptr<char[]> m_resource_buffer`：这是一个缓冲区，在将请求资源的内容作为响应消息的一部分发送到客户端之前，存储请求资源的内容

+   `unsigned int m_response_status_code`：这是 HTTP 响应状态码

+   `std::size_t m_resource_size_bytes`：这是请求资源的内容的尺寸

+   `std::string m_response_headers`：这是一个包含正确格式化的响应头块的字符串

+   `std::string m_response_status_line`：这包含一个响应状态行

现在我们知道了`Service`类数据成员的目的，让我们追踪它是如何工作的。在这里，我们只考虑`Service`类的工作方式。关于服务器应用程序的所有其他组件及其工作方式的描述，请参阅第四章中名为*实现异步 TCP 服务器*的配方，*实现服务器应用程序*。

当客户端发送 TCP 连接请求并且该请求在服务器上被接受（这发生在`Acceptor`类中，在本食谱中不予考虑）时，会创建`Service`类的一个实例，并将指向连接到该客户端的 TCP 套接字对象的共享指针传递给其构造函数。套接字指针存储在`Service`对象的数据成员`m_sock`中。

此外，在构造`Service`对象期间，`m_request`流缓冲区成员变量被初始化为 4096，这设置了缓冲区的最大字节数。限制请求缓冲区的大小是一种安全措施，有助于保护服务器免受可能尝试发送非常长的虚拟请求消息并耗尽服务器应用程序可用内存的恶意客户端。对于正确的请求，4096 字节的缓冲区大小绰绰有余。

在构造`Service`类的一个实例之后，`Acceptor`类会调用其`start_handling()`方法。从这个方法开始，异步方法调用的序列开始执行，该序列执行请求接收、处理和响应发送。`start_handling()`方法立即启动一个异步读取操作，调用`asio::async_read_until()`函数以接收客户端发送的 HTTP 请求行。`on_request_line_received()`方法被指定为回调。

当调用`on_request_line_received()`方法时，我们首先检查指定操作完成状态的错误代码。如果状态码不等于零，我们考虑两种选项。第一种选项——当错误代码等于`asio::error::not_found`值时——意味着从客户端接收的字节数超过了缓冲区的大小和 HTTP 请求行（`\r\n`符号序列）的分隔符尚未遇到。这种情况由 HTTP 状态码 413 描述。我们将`m_response_status_code`成员变量的值设置为 413，并调用`send_response()`方法，该方法启动向客户端发送指定错误响应的操作。我们将在本节稍后考虑`send_response()`方法。此时，请求处理已完成。

如果错误代码既不表示成功也不等于`asio::error::not_found`，这意味着发生了其他我们无法恢复的错误，因此，我们只是输出有关错误的信息，并且根本不向客户端回复。调用`on_finish()`方法进行清理，并中断与客户端的通信。

最后，如果 HTTP 请求行的接收成功，它将被解析以提取 HTTP 请求方法、标识请求资源的 URI 以及 HTTP 协议版本。因为我们的示例服务器只支持 `GET` 方法，如果请求行中指定的方法与 `GET` 不同，则进一步请求处理将被中断，并向客户端发送包含错误代码 501 的响应，告知客户端请求中指定的方法不被服务器支持。

同样，客户端在 HTTP 请求行中指定的 HTTP 协议版本将被检查，以确保它被服务器支持。因为我们的服务器应用程序只支持版本 1.1，如果客户端指定的版本不同，则向客户端发送包含 HTTP 状态代码 505 的响应，并中断请求处理。

从 HTTP 请求行中提取的 URI 字符串存储在 `m_requested_resource` 数据成员中，并将随后使用。

当接收到并解析 HTTP 请求行时，我们继续读取请求消息，以便读取请求头部块。为此，调用 `asio::async_read_until()` 函数。因为请求头部块以 `\r\n\r\n` 符号序列结束，所以这个符号序列被作为分隔符参数传递给函数。指定 `on_headers_received()` 方法作为操作完成回调。

`on_headers_received()` 方法执行类似于 `on_request_line_received()` 方法中执行的错误检查。在出现错误的情况下，请求处理中断。在成功的情况下，解析 HTTP 请求头部块并将其分解为单独的名称-值对，然后存储在 `m_request_headers` 成员映射中。在解析头部块之后，依次调用 `process_request()` 和 `send_response()` 方法。

`process_request()` 方法的目的是读取请求中指定的作为 URI 的文件，并将其内容放入缓冲区，然后从该缓冲区发送内容到客户端，作为响应消息的一部分。如果指定的文件在服务器根目录中找不到，则将 HTTP 状态代码 404（页面未找到）作为响应消息的一部分发送给客户端，并中断请求处理。

然而，如果找到请求的文件，首先计算其大小，然后在空闲内存中分配相应大小的缓冲区，并在该缓冲区中读取文件内容。

在此之后，将一个名为 *content-length* 的 HTTP 头部添加到 `m_response_headers` 字符串数据成员中，该头部指定了响应体的尺寸。此数据成员代表响应头部块，其值将随后作为响应消息的一部分使用。

到这一点，构建 HTTP 响应消息所需的所有成分都已可用，我们可以继续准备并发送响应给客户端。这是在`send_response()`方法中完成的。

`send_response()`方法从关闭套接字的接收端开始，让客户端知道服务器将不再从它那里读取任何数据。然后，它从`http_status_table`静态表中提取与存储在`m_response_status_code`成员变量中的状态代码相对应的响应状态消息。

接下来，构建 HTTP 响应状态行，并根据 HTTP 协议将头部块附加带有分隔符号序列`\r\n`。到这一点，响应消息的所有组件——响应状态行、响应头部块和响应体——都已准备好发送给客户端。这些组件以缓冲区向量的形式组合，每个缓冲区由`asio::const_buffer`类的实例表示，并包含响应消息的一个组件。缓冲区向量体现了一个由三部分组成的复合缓冲区。当这个复合缓冲区构建时，它被传递给`asio::async_write()`函数以发送给客户端。`Service`类的`on_response_sent()`方法被指定为回调。

当响应消息发送并且调用`on_response_sent()`回调方法时，它首先检查错误代码，如果操作失败则输出日志消息；然后关闭套接字并调用`on_finish()`方法。`on_finish()`方法反过来删除在调用它的上下文中`Service`对象的实例。

到这一点，客户端处理已经完成。

## 参见

+   来自第四章的*实现异步 TCP 服务器*配方，*实现服务器应用程序*，提供了更多关于如何实现作为本配方基础的异步 TCP 服务器的信息。

+   来自第六章的*使用定时器*配方，*其他主题*，展示了如何使用 Boost.Asio 提供的定时器。定时器可以用来实现异步操作超时机制。

# 为客户端应用程序添加 SSL/TLS 支持

客户端应用程序通常使用 SSL/TLS 协议发送敏感数据，如密码、信用卡号码、个人信息。SSL/TLS 协议允许客户端验证服务器并加密数据。服务器的验证允许客户端确保数据将被发送到预期的接收者（而不是恶意者）。数据加密保证即使传输的数据在途中被拦截，拦截者也无法使用它。

本食谱演示了如何使用 Boost.Asio 和 OpenSSL 库实现一个支持 SSL/TLS 协议的同步 TCP 客户端应用程序。本食谱中演示的 TCP 客户端应用程序名为 *实现同步 TCP 客户端*，来自 第三章，*实现客户端应用程序*，将其作为本食谱的基础，并对它进行了某些代码更改和添加，以便添加对 SSL/TLS 协议的支持。与同步 TCP 客户端的基础实现不同的代码被 *突出显示*，以便与 SSL/TLS 支持相关的代码能更好地与其他代码区分开来。

## 准备工作…

在开始本食谱之前，必须安装 OpenSSL 库，并将项目链接到它。有关库安装或项目链接的步骤超出了本书的范围。有关更多信息，请参阅 OpenSSL 库文档。

此外，由于本食谱基于另一个名为 *实现同步 TCP 客户端* 的食谱，来自 第三章，*实现客户端应用程序*，强烈建议在继续之前熟悉它。

## 如何做到这一点…

以下代码示例演示了支持 SSL/TLS 协议以验证服务器并加密传输数据的同步 TCP 客户端应用程序的可能实现。

我们从添加 `include` 和 `using` 指令开始我们的应用程序：

```cpp
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <iostream>

using namespace boost;
```

`<boost/asio/ssl.hpp>` 头文件包含提供与 OpenSSL 库集成的类型和函数。

接下来，我们定义一个扮演同步 SSL/TLS 启用 TCP 客户端角色的类：

```cpp
class SyncSSLClient {
public:
  SyncSSLClient(const std::string& raw_ip_address,
    unsigned short port_num) :
    m_ep(asio::ip::address::from_string(raw_ip_address),
    port_num),
    m_ssl_context(asio::ssl::context::sslv3_client),    
    m_ssl_stream(m_ios, m_ssl_context)
  {
 // Set verification mode and designate that 
 // we want to perform verification.
 m_ssl_stream.set_verify_mode(asio::ssl::verify_peer);

 // Set verification callback. 
 m_ssl_stream.set_verify_callback(this->bool{
 return on_peer_verify(preverified, context);
 });  
  }

  void connect() {
 // Connect the TCP socket.
 m_ssl_stream.lowest_layer().connect(m_ep);

 // Perform the SSL handshake.
 m_ssl_stream.handshake(asio::ssl::stream_base::client);
  }

  void close() {
    // We ignore any errors that might occur
 // during shutdown as we anyway can't
 // do anything about them.
 boost::system::error_code ec;

 m_ssl_stream.shutdown(ec); // Shutdown SSL.

 // Shut down the socket.
 m_ssl_stream.lowest_layer().shutdown(
 boost::asio::ip::tcp::socket::shutdown_both, ec);

 m_ssl_stream.lowest_layer().close(ec);
  }

  std::string emulate_long_computation_op(
    unsigned int duration_sec) {

    std::string request = "EMULATE_LONG_COMP_OP "
      + std::to_string(duration_sec)
      + "\n";

    send_request(request);
    return receive_response();
  };

private:
  bool on_peer_verify(bool preverified,
 asio::ssl::verify_context& context) 
 {
 // Here the certificate should be verified and the
 // verification result should be returned.
 return true;
 }

  void send_request(const std::string& request) {
    asio::write(m_ssl_stream, asio::buffer(request));
  }

  std::string receive_response() {
    asio::streambuf buf;
    asio::read_until(m_ssl_stream, buf, '\n');

    std::string response;
    std::istream input(&buf);
    std::getline(input, response);

    return response;
  }

private:
  asio::io_service m_ios;
  asio::ip::tcp::endpoint m_ep;

  asio::ssl::context m_ssl_context;
 asio::ssl::stream<asio::ip::tcp::socket>m_ssl_stream;
};
```

现在我们实现 `main()` 应用程序入口点函数，它使用 `SyncSSLClient` 类通过 SSL/TLS 协议验证服务器并与其安全通信：

```cpp
int main()
{
  const std::string raw_ip_address = "127.0.0.1";
  const unsigned short port_num = 3333;

  try {
    SyncSSLClient client(raw_ip_address, port_num);

    // Sync connect.
    client.connect();

    std::cout << "Sending request to the server... "
      << std::endl;

    std::string response =
      client.emulate_long_computation_op(10);

    std::cout << "Response received: " << response
      << std::endl;

    // Close the connection and free resources.
    client.close();
  }
  catch (system::system_error &e) {
    std::cout << "Error occured! Error code = " << e.code()
      << ". Message: " << e.what();

    return e.code().value();
  }

  return 0;
}
```

## 它是如何工作的…

示例客户端应用程序由两个主要组件组成：`SyncSSLClient` 类和 `main()` 应用程序入口点函数，后者使用 `SyncSSLClient` 类通过 SSL/TLS 协议与服务器应用程序通信。让我们分别考虑每个组件的工作原理。

### SyncSSLClient 类

`SyncSSLClient` 类是我们应用程序中的关键组件。它实现了通信功能。

该类有四个私有数据成员，如下所示：

+   `asio::io_service m_ios`：这是一个提供对操作系统通信服务的对象，这些服务由套接字对象使用。

+   `asio::ip::tcp::endpoint m_ep`：这是一个指定服务器应用程序的端点。

+   `asio::ssl::context m_ssl_context`：这是一个表示 SSL 上下文的对象；基本上，这是一个围绕 OpenSSL 库中定义的`SSL_CTX`数据结构包装的包装器。此对象包含用于通过 SSL/TLS 协议进行通信的其他对象和函数的全球设置和参数。

+   `asio::ssl::stream<asio::ip::tcp::socket> m_ssl_stream`：这表示一个包装 TCP 套接字对象的流，并实现了所有 SSL/TLS 通信操作。

类中的每个对象都旨在与单个服务器进行通信。因此，类的构造函数接受一个 IP 地址和一个协议端口号作为输入参数，指定服务器应用程序。这些值用于在构造函数的初始化列表中实例化`m_ep`数据成员。

接下来，实例化`SyncSSLClient`类的`m_ssl_context`和`m_ssl_stream`成员。我们将`asio::ssl::context::sslv23_client`值传递给`m_ssl_context`对象的构造函数，以指定上下文将仅由充当*客户端*的应用程序使用，并且我们希望支持包括 SSL 和 TLS 多个版本的多个安全协议。此由 Boost.Asio 定义的值对应于 OpenSSL 库中定义的`SSLv23_client_method()`函数返回的连接方法表示的值。

SSL 流对象`m_ssl_stream`在`SyncSSLClient`类的构造函数中设置。首先，将对方验证模式设置为`asio::ssl::verify_peer`，这意味着我们希望在握手过程中执行对方验证。然后，我们设置一个验证回调方法，该方法将在从服务器收到证书时被调用。对于服务器发送的证书链中的每个证书，回调都会被调用一次。

类的`on_peer_verify()`方法被设置为对方验证回调是一个虚拟的。证书验证过程超出了本书的范围。因此，该函数简单地始终返回`true`常量，这意味着证书验证成功，而没有执行实际的验证。

三个公共方法构成了`SyncSSLClient`类的接口。名为`connect()`的方法执行两个操作。首先，将 TCP 套接字连接到服务器。SSL 流对象底层的套接字由 SSL 流对象的`lowest_layer()`方法返回。然后，使用`m_ep`作为参数（指定要连接的端点）调用套接字上的`connect()`方法：

```cpp
// Connect the TCP socket.
m_ssl_stream.lowest_layer().connect(m_ep);

```

在 TCP 连接建立后，在 SSL 流对象上调用`handshake()`方法，这将导致握手过程的启动。此方法是同步的，直到握手完成或发生错误才返回：

```cpp
// Perform the SSL handshake.
m_ssl_stream.handshake(asio::ssl::stream_base::client);

```

在 `handshake()` 方法返回后，TCP 和 SSL（或 TLS，具体取决于握手过程中商定的协议）连接都建立，并且可以执行有效通信。

`close()` 方法通过在 SSL 流对象上调用 `shutdown()` 方法来关闭 SSL 连接。`shutdown()` 方法是同步的，并且会阻塞，直到 SSL 连接关闭或发生错误。在此方法返回后，相应的 SSL 流对象不能再用于传输数据。

第三个接口方法是 `emulate_long_computation_op(unsigned int duration_sec)`。这个方法是在这里执行 I/O 操作的地方。它从根据应用层协议准备请求字符串开始。然后，请求被传递到类的 `send_request(const std::string& request)` 私有方法，该方法将其发送到服务器。当请求发送并且 `send_request()` 方法返回时，调用 `receive_response()` 方法从服务器接收响应。当收到响应时，`receive_response()` 方法返回包含响应的字符串。之后，`emulate_long_computation_op()` 方法将响应消息返回给其调用者。

注意，`emulate_long_computation_op()`、`send_request()` 和 `receive_response()` 方法几乎与在 `SyncTCPClient` 类中定义的相应方法相同，`SyncTCPClient` 类是同步 TCP 客户端应用程序的一部分，该应用程序在第三章，*实现客户端应用程序*中演示，我们将其用作 `SyncSSLClient` 类的基础。唯一的区别是，在 `SyncSSLClient` 中，将 *SSL 流对象* 传递给相应的 Boost.Asio I/O 函数，而在 `SyncTCPClient` 类中，将这些函数传递给 *套接字对象*。提到的其他方法方面是相同的。

### `main()` 入口点函数

此函数充当 `SyncSSLClient` 类的用户。在获取服务器 IP 地址和协议端口号后，它实例化并使用 `SyncSSLClient` 类的对象来验证并安全地与服务器通信，以使用其服务，即通过执行 10 秒的虚拟计算来模拟服务器上的操作。此函数的代码简单且易于理解；因此，不需要额外的注释。

## 参见

+   第三章，*实现客户端应用程序*中的*实现同步 TCP 客户端*配方提供了有关如何实现作为本配方基础的同步 TCP 客户端的更多信息。

# 为服务器应用程序添加 SSL/TLS 支持

当服务器提供的服务假设客户端将敏感数据（如密码、信用卡号、个人信息等）传输到服务器时，通常会在服务器应用程序中添加 SSL/TLS 协议支持。在这种情况下，向服务器添加 SSL/TLS 协议支持允许客户端验证服务器，并建立一个安全通道，以确保在传输过程中敏感数据得到保护。

有时，服务器应用程序可能想使用 SSL/TLS 协议来验证客户端；然而，这种情况很少见，通常使用其他方法来确保客户端的真实性（例如，在登录邮件服务器时指定用户名和密码）。

本菜谱演示了如何使用 Boost.Asio 和 OpenSSL 库实现一个支持 SSL/TLS 协议的同步迭代 TCP 服务器应用程序。菜谱中演示的同步迭代 TCP 服务器应用程序名为 *实现同步迭代 TCP 服务器*，来自 第四章，*实现服务器应用程序*，本菜谱以此为基础，并对它进行了一些代码更改和添加，以便添加对 SSL/TLS 协议的支持。与同步迭代 TCP 服务器的基础实现不同的代码被 *突出显示*，以便与代码的其他部分更好地区分开来，直接相关的代码与 SSL/TLS 支持相关。

## 准备工作…

在开始本菜谱之前，必须安装 OpenSSL 库，并将项目链接到它。有关库安装或项目链接的步骤超出了本书的范围。有关更多信息，请参阅官方 OpenSSL 文档。

此外，因为本菜谱基于另一个名为 *实现同步迭代 TCP 服务器* 的菜谱，来自 第四章，*实现服务器应用程序*，所以在继续之前，强烈建议熟悉它。

## 如何实现…

以下代码示例演示了支持 SSL/TLS 协议的同步 TCP 服务器应用程序的可能实现，允许客户端验证服务器并保护传输中的数据。

我们从包含 Boost.Asio 库头文件以及我们将需要在应用程序中实现的一些标准 C++ 库组件的头文件开始我们的应用程序：

```cpp
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>

#include <thread>
#include <atomic>
#include <iostream>

using namespace boost;
```

`<boost/asio/ssl.hpp>` 头文件包含提供与 OpenSSL 库集成的类型和函数。

接下来，我们定义一个类，该类通过读取请求消息、处理它，然后发送响应消息来处理单个客户端。这个类代表服务器应用程序提供的一个单一服务，并相应地命名为 `Service`：

```cpp
class Service {
public:
  Service(){}

  void handle_client(
  asio::ssl::stream<asio::ip::tcp::socket>& ssl_stream) 
  {
    try {
      // Blocks until the handshake completes.
 ssl_stream.handshake(
 asio::ssl::stream_base::server);

      asio::streambuf request;
      asio::read_until(ssl_stream, request, '\n');

      // Emulate request processing.
      int i = 0;
      while (i != 1000000)
        i++;
      std::this_thread::sleep_for(
        std::chrono::milliseconds(500));

      // Sending response.
      std::string response = "Response\n";
      asio::write(ssl_stream, asio::buffer(response));
    }
    catch (system::system_error &e) {
      std::cout << "Error occured! Error code = "
        << e.code() << ". Message: "
        << e.what();
    }
  }
};
```

接下来，我们定义另一个代表高级 *接受者* 概念的类（与由`asio::ip::tcp::acceptor`类表示的低级接受者相比）。这个类负责接受来自客户端的连接请求并实例化`Service`类的对象，该对象将为连接的客户端提供服务。这个类被称为`Acceptor`:

```cpp
class Acceptor {
public:
  Acceptor(asio::io_service& ios, unsigned short port_num) :
    m_ios(ios),
    m_acceptor(m_ios,
    asio::ip::tcp::endpoint(
    asio::ip::address_v4::any(),
    port_num)),
    m_ssl_context(asio::ssl::context::sslv23_server)
  {
    // Setting up the context.
 m_ssl_context.set_options(
 boost::asio::ssl::context::default_workarounds
 | boost::asio::ssl::context::no_sslv2
 | boost::asio::ssl::context::single_dh_use);

 m_ssl_context.set_password_callback(
 this
 -> std::string 
 {return get_password(max_length, purpose);}
 );

 m_ssl_context.use_certificate_chain_file("server.crt");
 m_ssl_context.use_private_key_file("server.key",
 boost::asio::ssl::context::pem);
 m_ssl_context.use_tmp_dh_file("dhparams.pem");

    // Start listening for incoming connection requests.
    m_acceptor.listen();
  }

  void accept() {
    asio::ssl::stream<asio::ip::tcp::socket>
 ssl_stream(m_ios, m_ssl_context);

    m_acceptor.accept(ssl_stream.lowest_layer());

    Service svc;
    svc.handle_client(ssl_stream);
  }

private:
  std::string get_password(std::size_t max_length,
 asio::ssl::context::password_purpose purpose) const
 {
 return "pass";
 }

private:
  asio::io_service& m_ios;
  asio::ip::tcp::acceptor m_acceptor;

  asio::ssl::context m_ssl_context;
};
```

现在我们定义一个代表服务器本身的类。这个类被相应地命名为—`Server`:

```cpp
class Server {
public:
  Server() : m_stop(false) {}

  void start(unsigned short port_num) {
    m_thread.reset(new std::thread([this, port_num]() {
      run(port_num);
    }));
  }

  void stop() {
    m_stop.store(true);
    m_thread->join();
  }

private:
  void run(unsigned short port_num) {
    Acceptor acc(m_ios, port_num);

    while (!m_stop.load()) {
      acc.accept();
    }
  }

  std::unique_ptr<std::thread> m_thread;
  std::atomic<bool> m_stop;
  asio::io_service m_ios;
};
```

最终，我们实现了`main()`应用程序入口点函数，该函数演示了如何使用`Server`类。此函数与我们在第四章中定义的配方中的函数相同：

```cpp
int main()
{
  unsigned short port_num = 3333;

  try {
    Server srv;
    srv.start(port_num);

    std::this_thread::sleep_for(std::chrono::seconds(60));

    srv.stop();
  }
  catch (system::system_error &e) {
    std::cout   << "Error occured! Error code = " 
    << e.code() << ". Message: "
        << e.what();
  }

  return 0;
}
```

注意，服务器应用程序的最后两个组件，即`Server`类和`main()`应用程序入口点函数，与我们在第四章中定义的相应组件相同，该组件是我们为这个配方所采用的基。

## 它是如何工作的…

示例服务器应用程序由四个组件组成：`Service`、`Acceptor`和`Server`类以及`main()`应用程序入口点函数，它演示了如何使用`Server`类。由于`Server`类和`main()`入口点函数的源代码和目的与我们在第四章中定义的相应组件相同，该组件是我们为这个配方所采用的基，我们在这里将不讨论它们。我们只考虑更新以支持 SSL/TLS 协议的`Service`和`Acceptor`类。

### 服务类

`Service`类是应用程序中的关键功能组件。虽然其他组件在其目的上是基础设施性的，但这个类实现了客户端所需的实际功能（或服务）。

`Service`类相当简单，仅包含一个方法`handle_client()`。作为其输入参数，此方法接受一个表示封装了连接到特定客户端的 TCP 套接字的 SSL 流对象的引用。

方法从通过在`ssl_stream`对象上调用`handshake()`方法执行 SSL/TLS **握手**开始。此方法是同步的，直到握手完成或发生错误，它不会返回。

在握手完成之后，从 SSL 流中同步读取一个请求消息，直到遇到新的换行 ASCII 符号 `\n`。然后，处理请求。在我们的示例应用程序中，请求处理非常简单且是模拟的，包括运行一个循环执行一百万次递增操作，然后让线程休眠半秒钟。之后，准备响应消息并发送回客户端。

Boost.Asio 函数和方法可能抛出的异常在 `handle_client()` 方法中被捕获和处理，不会传播到方法的调用者，这样，如果处理一个客户端失败，服务器仍然可以继续工作。

注意，`handle_client()` 方法与我们在本章作为此菜谱基础的 第四章 中定义的 *实现一个同步迭代 TCP 服务器* 菜谱中定义的相应方法非常相似。不同之处在于，在这个菜谱中，`handle_client()` 方法操作一个代表 SSL 流的对象，而不是在方法的基本实现中操作代表 TCP 套接字的对象。此外，在这个菜谱中定义的方法还执行了一个额外的操作——SSL/TLS 握手。

### 接受者类

`Acceptor` 类是服务器应用程序基础设施的一部分。这个类的每个对象都拥有一个名为 `m_ssl_context` 的 `asio::ssl::context` 类的实例。这个成员代表一个 **SSL 上下文**。基本上，`asio::ssl::context` 类是 OpenSSL 库中定义的 `SSL_CTX` 数据结构的包装器。这个类的对象包含用于 SSL/TLS 协议通信过程中其他对象和函数的全局设置和参数。

当 `m_ssl_context` 对象被实例化时，其构造函数传入一个 `asio::ssl::context::sslv23_server` 值，以指定 SSL 上下文仅由充当 *服务器* 角色的应用程序使用，并且应该支持多个安全协议，包括 SSL 和 TLS 的多个版本。这个由 Boost.Asio 定义的值对应于 OpenSSL 库中定义的 `SSLv23_server_method()` 函数返回的连接方法值。

SSL 上下文在 `Acceptor` 类的构造函数中进行配置。上下文选项、密码回调以及包含数字证书、私钥和 Diffie-Hellman 协议参数的文件都在那里指定。

在配置 SSL 上下文之后，在 `Acceptor` 类的构造函数中调用 `listen()` 方法，以便开始监听来自客户端的连接请求。

`Acceptor` 类公开了一个单一的 `accept()` 公共方法。当调用此方法时，首先实例化一个名为 `ssl_stream` 的 `asio::ssl::stream<asio::ip::tcp::socket>` 类的对象，代表与底层 TCP 套接字的 SSL/TLS 通信通道。然后，在 `m_acceptor` 接收器对象上调用 `accept()` 方法以接受一个连接。`ssl_stream` 的 `lowest_layer()` 方法返回的拥有 TCP 套接字的对象作为输入参数传递给 `accept()` 方法。当建立新的连接时，创建 `Service` 类的一个实例，并调用其 `handle_client()` 方法，该方法执行与客户端的通信和请求处理。

## 参见

+   来自 第四章 的 *实现同步迭代 TCP 服务器* 菜谱，*实现服务器应用程序*，提供了更多关于如何实现作为本菜谱基础的同步 TCP 服务器的信息。
