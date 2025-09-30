# Chapter 5. HTTP and SSL/TLS

In this chapter, we will cover the following topics:

*   Implementing the HTTP client application
*   Implementing the HTTP server application
*   Adding SSL/TLS support to client applications
*   Adding SSL/TLS support to server applications

# Introduction

This chapter covers two major topics. The first one is HTTP protocol implementation. The second is the usage of SSL/TLS protocol. Let's briefly examine each of them.

The **HTTP protocol** is an application layer protocol operating on the top of TCP protocol. It is widely used on the Internet, allowing client applications to request particular resources from the servers, and servers to transmit the requested resources back to the clients. Besides, HTTP allows clients to upload data and send commands to the server.

The HTTP protocol assumes several models or **methods** of communication, each designed for a specific purpose. The simplest method called `GET` assumes the following flow of events:

1.  The HTTP client application (for example, a web browser) generates a request message containing information about a particular resource (residing on the server) to be requested and sends it to the HTTP server application (for example, a web server) using TCP as a transport level protocol.
2.  The HTTP server application, having received a request from the client, parses it, extracts the requested resource from the storage (for example, from a file system or a database), and sends it back to the client as a part of a HTTP response message.

The format of both the request and response messages is defined by HTTP protocol.

Several other methods are defined by HTTP protocol, allowing client application to actively send data or upload resources to the server, delete resources located on the server, and perform other operations. In the recipes of this chapter, we will consider implementation of the `GET` method. Because HTTP protocol methods are similar in principle, implementation of one of them gives a good hint about how to implement others.

Another topic covered in this chapter is **SSL and TLS protocols**. **Secure Socket Layer** (**SSL**)and **Transport Layer Security** (**TLS**) protocols operate on the top of TCP protocol and are aimed at achieving two main goals as follows:

*   Providing a way to authenticate each communication participant using digital certificate
*   Securing data being transmitted over the underlying TCP protocol

The SSL and TLS protocols are widespread, especially in the Web. Most web servers to which its potential clients may send sensitive data (passwords, credit card numbers, personal data, and so on) support SSL/TLS-enabled communication. In this case, the so called HTTPS (HTTP over SSL) protocol is used to allow the client to authenticate the server (sometimes servers may want to authenticate the client, though this is rarely the case) and to secure transmitted data by encrypting it, making this data useless for the culprit even if intercepted.

### Note

Boost.Asio does not contain the implementation of SSL/TLS protocols. Instead, it relies on the OpenSSL library, Boost.Asio provides a set of classes, functions, and data structures that facilitate the usage of functionality provided by OpenSSL, making the code of the application more uniformed and object-oriented.

In this chapter, we will not consider the details of the OpenSSL library or SSL/TLS protocols. These topics are not in the scope of this book. Instead, we will touch upon specific tools provided by the Boost.Asio that rely on OpenSSL library and allow to implement support of SSL/TLS protocol in a network application.

The two recipes demonstrate how to build client and server applications that secure their communication using SSL/TLS protocols. To make SSL/TLS-related aspects of the applications more vivid and clear, all other aspects of considered applications were made as simple as possible. Both client and server applications are synchronous and based on recipes found in other chapters of this book. This allows us to compare a basic TCP client or server application with their advanced versions supporting SSL/TLS and to better understand what it takes to add SSL/TLS support to a distributed application.

# Implementing the HTTP client application

HTTP clients constitute important class of distributed software and are represented by many applications. Web browsers are prominent representatives of this class. They use HTTP protocols to request web pages from web servers. However, today HTTP protocol is used not only in the web. Many distributed applications use this protocol to exchange custom data of any kind. Often, when designing a distributed application, choosing HTTP as a communication protocol is a much better idea than developing custom one.

In this recipe, we will consider an implementation of HTTP client using Boost.Asio that satisfies the following basic requirements:

*   Supports the HTTP `GET` request method
*   Executes requests asynchronously
*   Supports request canceling

Let's move on to the implementation.

## How to do it…

Because one of the requirements of our client application is to support canceling requests that have been initiated but have not been completed yet, we need to make sure that canceling is enabled on all target platforms. Therefore, we begin our client application by configuring Boost.Asio library so that request canceling is enabled. More details on issues related to asynchronous operation canceling are provided in the *Cancelling asynchronous operations* recipe in [Chapter 2](ch02.html "Chapter 2. I/O Operations"), *I/O Operations*:

[PRE0]

Next, we include Boost.Asio library headers and also headers of some components of standard C++ libraries that we will need to implement our application:

[PRE1]

Now, before we can jump to implementing classes and functions constituting our client application, we have to make one more preparation related to error representation and handling.

When implementing the HTTP client application, we need to deal with three classes of errors. The first class is represented by numerous errors that may occur when executing Boost.Asio functions and classes' methods. For example, if we call the `write_some()` method on an object representing a socket that has not been opened, the method will return operating system dependent error code (either by throwing an exception or by the means of an out argument depending on the method overload used), designating the fact that an invalid operation has been executed on a non-opened socket.

The second class includes both erroneous and non-erroneous statuses defined by HTTP protocol. For example, the status code 200 returned by the server as a response to particular request made by the client, designates the fact that a client's request has been fulfilled successfully. On the other hand, the status code 500 designates that while performing the requested operation, an error occurred on the server that led to the request not being fulfilled.

The third class includes errors related to the HTTP protocol itself. In case a server sends a message, as a response to correct the request made by a client and this message is not a properly structured HTTP response, the client application should have means to represent this fact in terms of error code.

Error code for the first class of errors are defined in the sources of Boost.Asio libraries. Status codes of the second class are defined by HTTP protocol. The third class is not defined anywhere and we should define corresponding error codes by ourselves in our application.

We define a single error code that represents quite a general error designating the fact that the message received from the server is not a correct HTTP response message and therefore, the client cannot parse it. Let's name this error code as `invalid_response`:

[PRE2]

Then, we define a class representing an error category, which includes the `invalid_response` error code defined above. Let's name this category as `http_errors_category`:

[PRE3]

Then, we define a static object of this class, a function returning an instance of the object, and the overload for the `make_error_code()` function accepting error codes of our custom type `http_error_codes`:

[PRE4]

The last step we need to perform before we can use our new error code in our application is to allow Boost library to *know* that the members of the `http_error_codes` enumeration should be treated as error codes. To do this, we include the following structure definition into the `boost::system` namespace:

[PRE5]

Because our HTTP client application is going to be asynchronous, the user of the client when initiating a request, will need to provide a pointer to a callback function, which will be invoked when the request completes. We need to define a type representing a pointer to such a callback function.

A callback function when called, would need to be passed arguments that clearly designate three things:

*   Which request has completed
*   What is the response
*   Whether the request completed successfully and if not, the error code designating the error that occurred

Note that, later, we will define the `HTTPRequest` and `HTTPResponse` classes representing the HTTP request and HTTP response correspondingly, but now we use forward declarations. Here is how the callback function pointer type declaration looks:

[PRE6]

### The HTTPResponse class

Now, we can define a class representing a HTTP response message sent to the client as a response to the request:

[PRE7]

The `HTTPResponse` class is quite simple. Its private data members represent parts of HTTP response such as the response status code and status message, and response headers and body. Its public interface contains methods that return the values of corresponding data members, while private methods allow setting those values.

The `HTTPRequest` class representing a HTTP request, which will be defined next, is declared as a friend to `HTTPResponse`. We will see how the objects of the `HTTPRequest` class use the private methods of the `HTTPResponse` class to set values of its data members when a response message arrives.

### The HTTPRequest class

Next, we define a class representing a HTTP request containing functionality that constructs the HTTP request message based on information provided by the class user, sends it to the server, and then receives and parses the HTTP response message.

This class is at the center of our application because it contains most of its functionalities.

Later, we will define the `HTTPClient` class representing an HTTP client, responsibilities of which will be limited to maintaining a single instance of the `asio::io_service` class common to all the `HTTPRequest` objects and acting as a factory of the `HTTPRequest` objects. Therefore, we declare the `HTTPClient` class as a friend to the `HTTPRequest` class and make the `HTTPRequest` class' constructor private:

[PRE8]

The constructor accepts two arguments: a reference to an object of the `asio::io_service` class and an unsigned integer named `id`. The latter contains a unique identifier of a request, which is assigned by the user of the class and allows distinguishing request objects one from another.

Then, we define methods constituting the public interface of the class:

[PRE9]

The public interface includes methods that allow the class' user to set and get HTTP request parameters such as the DNS name of the host running the server, protocol port number, and URI of the requested resource. Besides, there is a method that allows setting a pointer to a callback function that will be called when the request completes.

The `execute()` method initiates the execution of the request. Also, the `cancel()` method allows canceling the initiated request before it completes. We will consider how these methods work in the next section of the recipe.

Now, we define a set of private methods that contain most of the implementation details. Firstly, we define a method that is used as a callback for an asynchronous DNS name resolution operation:

[PRE10]

Then, we define a method used as a callback for an asynchronous connection operation, which is initiated in the `on_host_name_resolved()` method just defined:

[PRE11]

The next method we define—`on_request_sent()`—is a callback, which is called after the request message is sent to the server:

[PRE12]

Then, we need another callback method, which is called when the first portion of the response message, namely, **status line**, is received from the server:

[PRE13]

Now, we define a method that serves as a callback, which is called when the next portion of the response message—**the** **response headers block**—arrives from the server. We will name it as `on_headers_received()`:

[PRE14]

Besides, we need a method that will handle the last part of the response—**the response body**. The following method is used as a callback, which is called after the response body arrives from the server:

[PRE15]

Finally, we define the `on_finish()` method that serves as a final point of all execution paths (including erroneous) that start in the `execute()` method. This method is called when the request completes (either successfully or not) and its purpose is to call the callback provided by the `HTTPRequest` class' user to notify it about the completion of the request:

[PRE16]

We will need some data fields associated with each instance of the `HTTPRequest` class. Here, we declare the class' corresponding data members:

[PRE17]

The last thing to add is the closing bracket to designate the end of the `HTTPRequest` class definition:

[PRE18]

### The HTTPClient class

The last class that we need in our application is the one that would be responsible for the following three functions:

*   To establish a threading policy
*   To spawn and destroy threads in a pool of threads running the Boost.Asio event loop and delivering asynchronous operations' completion events
*   To act as a factory of the `HTTPRequest` objects

We will name this class as `HTTPClient`:

[PRE19]

### The callback and the main() entry point function

At this point, we have the basic HTTP client that comprises three classes and several supplementary data types. Now we will define two functions that are not parts of the client, but demonstrate how to use it to communicate with the server using the HTTP protocol. The first function will be used as a callback, which will be called when the request completes. Its signature must correspond to the function pointer type `Callback` defined earlier. Let's name our callback function as `handler()`:

[PRE20]

The second and the last function we need to define is the `main()` application entry point function that uses the HTTP client to send HTTP requests to the server:

[PRE21]

## How it works…

Now let's consider how our HTTP client works. The application consists of five components, among which are the three classes such as `HTTPClient`, `HTTPRequest`, and `HTTPResponse`, and two functions such as the `handler()` callback function and the `main()` application entry point function. Let's consider how each component works separately.

### The HTTPClient class

A class' constructor begins with creating an instance of the `asio::io_service::work` object in order to make sure that threads running the event loop do not exit this loop when there are no pending asynchronous operations. Then, a thread of control is spawned and added to the pool by calling the `run()` method on the `m_ios` object. This is where the `HTTPClient` class performs its first and part of the second functions, namely, establishing threading policy and adding threads to the pool.

The third function of the `HTTPClient` class—to act as a factory of the object representing HTTP requests—is performed in its `create_request()` public method. This method creates an instance of the `HTTPRequest` class in the free memory and returns a shared pointer object pointing to it. As its input argument, the method accepts an integer value that represents the unique identifier to be assigned to the newly created request object. This identifier is used to distinguish between different request objects.

The `close()` method from the class' public interface destroys the `asio::io_service::work` object, allowing threads to exit the event loop just as soon as all pending operations complete. The method blocks until all threads exit.

### The HTTPRequest class

Let's begin considering the `HTTPRequest` class' behavior by inspecting its data members and their purpose. The `HTTPRequest` class contains 12 data members, among which are the following:

*   Request parameters:

    [PRE22]

*   A unique identifier of the request:

    [PRE23]

*   A pointer to the callback function provided by the class' user to be called when a request completes:

    [PRE24]

*   A string buffer used to store the HTTP request message:

    [PRE25]

*   A socket object used to communicate with the server:

    [PRE26]

*   A resolver object used to resolve the DNS name of the server host provided by the user:

    [PRE27]

*   An instance of the `HTTPResponse` class that represents the response received from the server:

    [PRE28]

*   A boolean flag and a `mutex` object supporting the request canceling functionality (which will be explained later):

    [PRE29]

*   Also, a reference to an instance of the `asio::io_service` class required by resolver and socket objects. The single instance of the `asio::io_service` class is maintained by an object of the `HTTPClient` class:

    [PRE30]

An instance of the `HTTPRequest` object represents a single HTTP `GET` request. The class is designed so that in order to send a request, two steps need to be performed. Firstly, the parameters of the request and the callback function to be called when the request completes are set by calling the corresponding setter methods on the object. Then, as a second step, the `execute()` method is invoked to initiate the request execution. When the request completes, the callback function is called.

The `set_host()`, `set_port()`, `set_uri()`, and `set_callback()` setter methods allow setting a server host DNS name and port number, URI of the requested resource, and a callback function to be called when the request completes. Each of these methods accepts one argument and stores its value in the corresponding `HTTPRequest` object's data member.

The `get_host()`, `get_port()`, and `get_uri()` getter methods return values set by corresponding setter methods. The `get_id()` getter method returns a request object's unique identifier, which is passed to the object's constructor on instantiation.

The `execute()` method begins the execution of a request by initiating a sequence of asynchronous operations. Each asynchronous operation performs one step of request execution procedure.

Because a server host in the request object is represented with a DNS name (rather than with an IP address), before sending the request message to the server, the specified DNS name must be resolved and transformed into an IP address. Therefore, the first step in the request execution is DNS name resolution. The `execute()` method begins with preparing the resolving query and then calls the resolver object's `async_resolve()` method, specifying the `HTTPRequest` class' `on_host_name_resolve()` private method as an operation completion callback.

When the server host DNS name is resolved, the `on_host_name_resolved()` method is called. This method is passed two arguments: the first of which is an error code, designating the status of the operation, and the second one is the iterator that can be used to iterate through a list of endpoints resulting from a resolution process.

The `on_host_name_resolved()` method initiates the next asynchronous operation in a sequence, namely socket connection, by calling `asio::async_connect()` free function passing socket object `m_sock` and iterator parameter to it so that it connects the socket to the first valid endpoint. The `on_connection_established()` method is specified as an asynchronous connection operation completion callback.

When an asynchronous connection operation completes, the `on_connection_established()` method is invoked. The first argument passed to it is named `ec` that designates the operation completion status. If its value is equal to zero, it means that the socket was successfully connected to one of the endpoints. The `on_connection_established()` method constructs the HTTP `GET` request message using request parameters stored in the corresponding data members of the `HTTPRequest` object. Then, the `asio::async_write()` free function is called to asynchronously send a constructed HTTP request message to the server. The class' private method `on_request_sent()` is specified as a callback to be called when the `asio::async_write()` operation completes.

After a request is sent, and if it is sent successfully, the client application has to let the server know that the full request is sent and the client is not going to send anything else by shutting down the send part of the socket. Then, the client has to wait for the response message from the server. And this is what the `on_request_sent()` method does. Firstly, it calls the socket object's `shutdown()` method, specifying that the send part should be closed by the passing value `asio::ip::tcp::socket::shutdown_send` to the method as an argument. Then, it calls the `asio::async_read_until()` free function to receive a response from the server.

Because the response may be potentially very big and we do not know its size beforehand, we do not want to read it all at once. We first want to read the **HTTP response status line** only; then, having analyzed it, either continue reading the rest of the response (if we think we need it) or discard it. Therefore, we pass the `\r\n` symbols sequence, designating the end of the HTTP response status line as a delimiter argument to the `asio::async_read_until()` method. The `on_status_line_received()` method is specified as an operation completion callback.

When the status line is received, the `on_status_line_received()` method is invoked. This method performs parsing of the status line, extracting values designating the HTTP protocol version, response status code, and response status message from it. Each value is analyzed for correctness. We expect the HTTP version to be 1.1, otherwise the response is considered incorrect and the request execution is interrupted. The status code should be an integer value. If the string-to-integer conversion fails, the response is considered incorrect and its further processing is interrupted too. If the response status line is correct, the request execution continues. The extracted status code and status message are stored in the `m_response` member object, and the next asynchronous operation in the request execution operation sequence is initiated. Now, we want to read the response headers block.

According to the HTTP protocol, the response headers block ends with the `\r\n\r\n` symbols sequence. Therefore, in order to read it, we call the `asio::async_read_until()` free function one more time, specifying the string `\r\n\r\n` as a delimiter. The `on_headers_received()` method is specified as a callback.

When the response headers block is received, the `on_headers_received()` method is invoked. In this method, the response headers block is parsed and broken into separate name-value pairs and stored in the `m_response` member object as a part of the response.

Having received and parsed the headers, we want to read the last part of the response—the response body. To do this, an asynchronous reading operation is initiated by calling the `asio::async_read()` free function. The `on_response_body_received()` method is specified as a callback.

Eventually, the `on_response_body_received()` method is invoked notifying us of the fact that the whole response message has been received. Because the HTTP server may shutdown the send part of its socket just after it sends the last part of the response message, on the client side, the last reading operation may complete with an error code equal to the `asio::error::eof` value. This should not be treated as an actual error, but rather as a normal event. Therefore, if the `on_response_body_received()` method is called with the `ec` argument equal to `asio::error::eof`, we pass the default constructed object of the `boost::system::error_code` class to the `on_finish()` method in order to designate that the request execution is completed successfully. Otherwise, the `on_finish()` method is called with an argument representing the original error code. The `on_finish()` method in its turn calls the callback provided by the client of the `HTTPRequest` class object.

When the callback returns, request processing is considered finished.

### The HTTPResponse class

The `HTTPResponse` class does not provide much functionality. It is more like a plain data structure containing data members representing different parts of a response, with getter and setter methods defined, allowing getting and setting corresponding data member values.

All setter methods are private and only the objects of the `HTTPRequest` class has access to them (recall that the `HTTPRequest` class is declared as the `HTTPResponse` class' friend). Each object of the `HTTPRequest` class has a data member that is an instance of the `HTTPResponse` class. The object of the `HTTPRequest` class sets values of its member object of `HTTPResponse` class as it receives and parses the response received from a HTTP server.

### Callback and the main() entry point functions

These functions demonstrate how to use the `HTTPClient` and `HTTPRequest` classes in order to send the `GET` HTTP requests to the HTTP server and then how to use the `HTTPResponse` class to obtain the response.

The `main()` function first creates an instance of the `HTTPClient` class and then uses it to create two instances of the `HTTPRequest` class, each representing a separate `GET` HTTP request. Both request objects are provided with request parameters and then executed. However, just after the second request has been executed, the first one is canceled by invoking its `cancel()` method.

The `handler()` function, which is used as a completion callback for both request objects created in the `main()` function, is invoked when each request completes regardless of whether it succeeded, failed, or was canceled. The `handler()` function analyses the error code and the request and response objects passed to it as arguments and output corresponding messages to the standard output stream.

## See also

*   The *Implementing asynchronous TCP client* recipe from [Chapter 3](ch03.html "Chapter 3. Implementing Client Applications"), *Implementing Client Applications*, provides more information on how to implement an asynchronous TCP client.
*   The *Using timers* recipe from [Chapter 6](ch06.html "Chapter 6. Other Topics"), *Other Topics*, demonstrates how to use timers provided by Boost.Asio. Timers can be used to implement an asynchronous operation timeout mechanism.

# Implementing the HTTP server application

Nowadays, there are plenty of HTTP server applications available in the market. However, sometimes there is a need to implement a custom one. This could be a small and simple server, supporting a specific subset of HTTP protocol possibly with custom extensions, or maybe not an HTTP server but a server supporting a communication protocol, which is similar to HTTP or is based on it.

In this recipe, we will consider the implementation of basic HTTP server application using Boost.Asio. Here is the set of requirements that our application must satisfy:

*   It should support the HTTP 1.1 protocol
*   It should support the `GET` method
*   It should be able to process multiple requests in parallel, that is, it should be an asynchronous parallel server

In fact, we have already considered the implementation of the server application that partially fulfils specified requirements. In [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, the recipe named *Implementing an asynchronous TCP server* demonstrates how to implement an asynchronous parallel TCP server, which communicates with clients according to a dummy application layer protocol. All the communication functionality and protocol details are encapsulated in a single class named `Service`. All other classes and functions defined in that recipe are infrastructural in their purpose and isolated from the protocol details. Therefore, the current recipe will be based on the one from [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, and here we will only consider the implementation of the `Service` class as all other components stay the same.

### Note

Note that, in this recipe, we do not consider the security aspect of the application. Make sure the server is protected before making it available to the public, where though operating correctly and in accordance with HTTP protocol, it could be compromised by the culprits due to security breaches.

Now let's move on to the implementation of the HTTP server application.

## Getting ready…

Because the application demonstrated in this recipe is based on other applications demonstrated in the recipe named *Implementing asynchronous TCP server* from [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, it is necessary to get acquainted with that recipe before proceeding with this one.

## How to do it…

We begin our application by including header files containing declarations and definitions of data types and functions that we will use:

[PRE31]

Next, we start defining the `Service` class that provides the implementation of the HTTP protocol. Firstly, we declare a static constant table containing HTTP status codes and status messages. The definition of the table will be given after the `Service` class' definition:

[PRE32]

The class' constructor accepts a single parameter—shared pointer pointing to an instance of a socket connected to a client. Here's the definition of the constructor:

[PRE33]

Next, we define a single method constituting the `Service` class' public interface. This method initiates an asynchronous communication session with the client connected to the socket, pointer to which was passed to the `Service` class' constructor:

[PRE34]

Then, we define a set of private methods that perform receiving and processing of the request sent by the client, parse and execute the request, and send the response back. Firstly, we define a method that processes the **HTTP request line**:

[PRE35]

Next, we define a method intended to process and store the **request headers block**, containing the request headers:

[PRE36]

Besides, we need a method that can perform the actions needed to fulfill the request sent by the client. We define the `process_request()` method, whose purpose is to read the contents of the requested resource from the file system and store it in the buffer, ready to be sent back to the client:

[PRE37]

Finally, we define a method that composes a response message and send it to the client:

[PRE38]

When the response sending is complete, we need to shut down the socket to let the client know that a full response has been sent and no more data will be sent by the server. We define the `on_response_sent()` method for this purpose:

[PRE39]

The last method we need to define is the one that performs cleanup and deletes an instance of the `Service` object, when the communication session is finished and the object is not needed anymore is not needed anymore:

[PRE40]

Of course, we will need some data members in our class. We declare the following data members:

[PRE41]

The last thing we need to do to complete the definition of the class representing a service is to define the `http_status_table` static member declared before and fill it with data—HTTP status code and corresponding status messages:

[PRE42]

Our `Service` class is now ready.

## How it works…

Let's begin with considering the `Service` class' data members and then switch to its functionality. The `Service` class contains the following non-static data members:

*   `std::shared_ptr<boost::asio::ip::tcp::socket> m_sock`: This is a shared pointer to a TCP socket object connected to the client
*   `boost::asio::streambuf m_request`: This is a buffer into which the request message is read
*   `std::map<std::string, std::string> m_request_headers`: This is a map where request headers are put when the HTTP request headers block is parsed
*   `std::string m_requested_resource`: This is the URI of the resource requested by the client
*   `std::unique_ptr<char[]> m_resource_buffer`: This is a buffer where the contents of a requested resource is stored before being sent to the client as a part of the response message
*   `unsigned int m_response_status_code`: This is the HTTP response status code
*   `std::size_t m_resource_size_bytes`: This is the size of the contents of the requested resource
*   `std::string m_response_headers`: This is a string containing a properly formatted response headers block
*   `std::string m_response_status_line`: This contains a response status line

Now that we know the purpose of the `Service` class' data members, let's trace how it works. Here, we will only consider how the `Service` class works. The description of all other components of the server application and how they work is given in the recipe named *Implementing an asynchronous TCP server* in [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*.

When a client sends a TCP connection request and this request is accepted on the server (this happens in the `Acceptor` class, which is not considered in this recipe), an instance of the `Service` class is created and its constructor is passed a shared pointer pointing to the TCP socket object, connected to that client. The pointer to the socket is stored in the `Service` object's data member `m_sock`.

Besides, during the construction of the `Service` object, the `m_request` stream buffer member is initialized with the value of 4096, which sets the maximum size of the buffer in bytes. Limiting the size of the request buffer is a security measure, which helps to protect the server from malicious clients that may try to send very long dummy request messages exhausting all memory at the disposal of the server application. For the correct request, a buffer of 4096 bytes in size is more than enough.

After an instance of the `Service` class has been constructed, its `start_handling()` method is called by the `Acceptor` class. From this method, the sequence of asynchronous method invocations begins, which performs request receiving, processing, and response sending. The `start_handling()` method immediately initiates an asynchronous reading operation calling the `asio::async_read_until()` function in order to receive the HTTP request line sent by the client. The `on_request_line_received()` method is specified as a callback.

When the `on_request_line_received()` method is invoked, we first check the error code specifying the operation completion status. If the status code is not equal to zero, we consider two options. The first option—when the error code is equal to the `asio::error::not_found` value—means that more bytes have been received from the client than the size of the buffer and the delimiter of the HTTP request line (the `\r\n` symbol sequence) has not been encountered. This case is described by the HTTP status code 413\. We set the value of the `m_response_status_code` member variable to 413 and call the `send_response()` method that initiates the operation that sends a response designating the error back to the client. We will consider the `send_response()` method later in this section. At this point, the request processing is finished.

If the error code neither designates success nor is equal to `asio::error::not_found`, it means that some other error has occurred from which we cannot recover, therefore, we just output the information about the error and do not reply to the client at all. The `on_finish()` method is called to perform the cleanup, and the communication with the client is interrupted.

Finally, if receiving of the HTTP request line succeeds, it is parsed to extract the HTTP request method, the URI identifying the requested resource and the HTTP protocol version. Because our sample server only supports the `GET` method, if the method specified in the request line is different from `GET`, further request processing is interrupted and the response containing the error code 501 is sent to the client to inform it that the method specified in the request is not supported by the server.

Likewise, the HTTP protocol version specified by the client in the HTTP request line is checked to be the one supported by the server. Because our server application supports only version 1.1, if the version specified by the client is different, the response with the HTTP status code 505 is sent to the client and the request processing is interrupted.

A URI string extracted from the HTTP request line is stored in the `m_requested_resource` data member and will be used later.

When the HTTP request line is received and parsed, we continue reading the request message in order to read the request headers block. To do this, the `asio::async_read_until()` function is called. Because the request headers block ends with the `\r\n\r\n` symbol sequence, this symbol sequence is passed to the function as a delimiter argument. The `on_headers_received()` method is specified as an operation completion callback.

The `on_headers_received()` method performs error checking similar to the one that is performed in the `on_request_line_received()` method. In case of an error, request processing interrupts. In the case of success, the HTTP request headers block is parsed and broken into separate name-value pairs, which are then stored in the `m_request_headers` member map. After the headers block has been parsed, the `process_request()` and `send_response()` methods are called consequently.

The purpose of the `process_request()` method is to read the file specified in the request as the URI and put its content to the buffer, from which the contents will be sent to the client as a part of the response message. If the specified file is not found in the server root directory, the HTTP status code 404 (page not found) code is sent to the client as a part of the response message and the request processing interrupts.

However, if the requested file is found, its size is first calculated and then the buffer of the corresponding size is allocated in the free memory and the file contents are read in that buffer.

After this, an HTTP header named *content-length* specifying the size of the response body is added to the `m_response_headers` string data member. This data member represents the response headers block and its value will later be used as a part of the response message.

At this point, all ingredients required to construct the HTTP response message are available and we can move on to preparing and sending the response to the client. This is done in the `send_response()` method.

The `send_response()` method starts with shutting down the receive side of the socket letting the client know that the server will not read any data from it anymore. Then, it extracts the response status message corresponding to the status code stored in the `m_response_status_code` member variable from the `http_status_table` static table.

Next, the HTTP response status line is constructed and the headers block is appended with the delimiting symbol sequence `\r\n` according to the HTTP protocol. At this point, all the components of the response message—the response status line, response headers block, and response body—are ready to be sent to the client. The components are combined in the form of a vector of buffers, each represented with an instance of the `asio::const_buffer` class and containing one component of the response message. A vector of buffers embodies a composite buffer consisting of three parts. When this composite buffer is constructed, it is passed to the `asio::async_write()` function to be sent to the client. The `Service` class' `on_response_sent()` method is specified as a callback.

When the response message is sent and the `on_response_sent()` callback method is invoked, it first checks the error code and outputs the log message if the operation fails; then, it shuts down the socket and calls the `on_finish()` method. The `on_finish()` method in its turn deletes the instance of the `Service` object in the context of which it is called.

At this point, client handling is finished.

## See also

*   The *Implementing an asynchronous TCP server* recipe from [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, provides more information on how to implement the asynchronous TCP server used as a base for this recipe.
*   The *Using timers* recipe from [Chapter 6](ch06.html "Chapter 6. Other Topics"), *Other Topics*, demonstrates how to use timers provided by Boost.Asio. Timers can be used to implement an asynchronous operation timeout mechanism.

# Adding SSL/TLS support to client applications

Client applications usually use SSL/TLS protocol to send sensitive data such as passwords, credit card numbers, personal data. SSL/TLS protocol allows clients to authenticate the server and encrypt the data. The authentication of the server allows the client to make sure that the data will be sent to the expected addressee (and not to a malicious one). Data encryption guarantees that even if the transmitted data is intercepted somewhere on its way to the server, the interceptor will not be able to use it.

This recipe demonstrates how to implement a synchronous TCP client application supporting SSL/TLS protocol using the Boost.Asio and OpenSSL libraries. The TCP client application demonstrated in the recipe named *Implementing synchronous TCP client* from [Chapter 3](ch03.html "Chapter 3. Implementing Client Applications"), *Implementing Client Applications*, is taken as a base for this recipe, and some code changes and additions are made to it in order to add support for SSL/TLS protocol. The code that differs from that of the base implementation of the synchronous TCP client is *highlighted* so that the code directly related to SSL/TLS support is better distinguished from the rest of the code.

## Getting ready…

Before setting out to this recipe, OpenSSL library must be installed and the project must be linked against it. Procedures related to the installation of the library or linking the project against it are beyond the scope of this book. Refer to the OpenSSL library documentation for more information.

Besides, because this recipe is based on another recipe named *Implementing a synchronous TCP Client* from [Chapter 3](ch03.html "Chapter 3. Implementing Client Applications"), *Implementing Client Applications*, it is highly advised to get acquainted with it before proceeding to this one.

## How to do it…

The following code sample demonstrates the possible implementation of a synchronous TCP client application supporting SSL/TLS protocol to authenticate the server and encrypt the data being transmitted.

We begin our application by adding the `include` and `using` directives:

[PRE43]

The `<boost/asio/ssl.hpp>` header contains types and functions providing integration with OpenSSL library.

Next, we define a class that plays the role of the synchronous SSL/TLS-enabled TCP client:

[PRE44]

Now we implement the `main()` application entry point function that uses the `SyncSSLClient` class to authenticate the server and securely communicate with it using SSL/TLS protocol:

[PRE45]

## How it works…

The sample client application consists of two main components: the `SyncSSLClient` class and a `main()` application entry point function that uses the `SyncSSLClient` class to communicate with the server application over SSL/TLS protocol. Let's consider how each component works separately.

### The SyncSSLClient class

The `SyncSSLClient` class is the key component in our application. It implements the communication functionality.

The class has four private data members as follows:

*   `asio::io_service m_ios`: This is an object providing access to the operating system's communication services that are used by the socket object.
*   `asio::ip::tcp::endpoint m_ep`: This is an endpoint designating the server application.
*   `asio::ssl::context m_ssl_context`: This is an object representing SSL context; basically, this is a wrapper around the `SSL_CTX` data structure defined by OpenSSL library. This object contains global settings and parameters used by other objects and functions involved in the process of communication using SSL/TLS protocol.
*   `asio::ssl::stream<asio::ip::tcp::socket> m_ssl_stream`: This represents a stream that wraps a TCP socket object and implements all SSL/TLS communication operations.

Each object of the class is intended to communicate with a single server. Therefore, the class' constructor accepts an IP address and a protocol port number designating the server application as its input arguments. These values are used to instantiate the `m_ep` data member in the constructor's initialization list.

Next, the `m_ssl_context` and `m_ssl_stream` members of the `SyncSSLClient` class are instantiated. We pass the `asio::ssl::context::sslv23_client` value to the `m_ssl_context` object's constructor to designate that the context will be used by the application playing a role of a *client* only and that we want to support multiple secure protocols including multiple versions of SSL and TLS. This value defined by Boost.Asio corresponds to a value representing a connection method returned by the `SSLv23_client_method()` function defined by OpenSSL library.

The SSL stream object `m_ssl_stream` is set up in the `SyncSSLClient` class' constructor. Firstly, the peer verification mode is set to `asio::ssl::verify_peer`, which means that we want to perform peer verification during a handshake. Then, we set a verification callback method that will be called when certificates arrive from the server. The callback is invoked once for each certificate in the certificates chain sent by the server.

The class' `on_peer_verify()` method that is set as a peer verification callback is a dummy in our application. The certificate verification process lies beyond the scope of this book. Therefore, the function simply always returns the `true` constant, meaning that the certificate verification succeeded without performing the actual verification.

The three public methods comprise the interface of the `SyncSSLClient` class. The method named `connect()` performs two operations. Firstly, the TCP socket is connected to the server. The socket underlying the SSL stream is returned by the method of the SSL stream object `lowest_layer()`. Then, the `connect()` method is called on the socket with `m_ep` being passed as an argument designating the endpoint to be connected to:

[PRE46]

After the TCP connection is established, the `handshake()` method is called on the SSL stream object, which leads to the initiation of the handshake process. This method is synchronous and does not return until the handshake completes or an error occurs:

[PRE47]

After the `handshake()` method returns, both TCP and SSL (or TLS, depending on which protocol was agreed upon during the handshake process) connections are established and the effective communication can be performed.

The `close()` method shuts down the SSL connection by calling the `shutdown()` method on the SSL stream object. The `shutdown()` method is synchronous and blocks until the SSL connection is shut down or an error occurs. After this method returns, the corresponding SSL stream object cannot be used to transmit the data anymore.

The third interface method is `emulate_long_computation_op(unsigned int duration_sec)`. This method is where the I/O operations are performed. It begins with preparing the request string according to the application layer protocol. Then, the request is passed to the class' `send_request(const std::string& request)` private method, which sends it to the server. When the request is sent and the `send_request()` method returns, the `receive_response()` method is called to receive the response from the server. When the response is received, the `receive_response()` method returns the string containing the response. After this, the `emulate_long_computation_op()` method returns the response message to its caller.

Note that the `emulate_long_computation_op()`, `send_request()`, and `receive_response()` methods are almost identical to the corresponding methods defined in the `SyncTCPClient` class, which is a part of the synchronous TCP client application demonstrated in [Chapter 3](ch03.html "Chapter 3. Implementing Client Applications"), *Implementing Client Applications*, which we used as a base for `SyncSSLClient` class. The only difference is that in `SyncSSLClient`, an *SSL stream object* is passed to the corresponding Boost.Asio I/O functions, while in the `SyncTCPClient` class, a *socket object* is passed to those functions. Other aspects of the mentioned methods are identical.

### The main() entry point function

This function acts as a user of the `SyncSSLClient` class. Having obtained the server IP address and protocol port number, it instantiates and uses the object of the `SyncSSLClient` class to authenticate and securely communicate with the server in order to consume its service, namely, to emulate an operation on the server by performing dummy calculations for 10 seconds. The code of this function is simple and self-explanatory; thus, requires no additional comments.

## See also

*   The *Implementing a synchronous TCP client* recipe from [Chapter 3](ch03.html "Chapter 3. Implementing Client Applications"), *Implementing Client Applications*, provides more information on how to implement a synchronous TCP client used as a base for this recipe.

# Adding SSL/TLS support to server applications

SSL/TLS protocol support is usually added to the server application when the services it provides assumes transmission of sensitive data such as passwords, credit card numbers, personal data, and so on, by the client to the server. In this case, adding SSL/TLS protocol support to the server allows clients to authenticate the server and establish a secure channel to make sure that the sensitive data is protected while being transmitted.

Sometimes, a server application may want to use SSL/TLS protocol to authenticate the client; however, this is rarely the case and usually other methods are used to ensure the authenticity of the client (for example, username and password are specified when logging into a mail server).

This recipe demonstrates how to implement a synchronous iterative TCP server application supporting SSL/TLS protocol using the Boost.Asio and OpenSSL libraries. The synchronous iterative TCP server application demonstrated in the recipe named *Implementing a synchronous iterative TCP server* from [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, is taken as a base for this recipe and some code changes and additions are made to it in order to add support for SSL/TLS protocol. The code that differs from that of the base implementation of the synchronous iterative TCP server is *highlighted* so that the code directly related to SSL/TLS support is better distinguished from the rest of the code.

## Getting ready…

Before setting out to this recipe, OpenSSL library must be installed and the project must be linked against it. Procedures related to the installation of the library or linking the project against it are beyond the scope of this book. Refer to the official OpenSSL documentation for more information.

Besides, because this recipe is based on another recipe named *Implementing a synchronous iterative TCP server*, from [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, it is highly advised to get acquainted with it before proceeding to this one.

## How to do it…

The following code sample demonstrates the possible implementation of a synchronous TCP server application supporting SSL/TLS protocol to allow client applications to authenticate the server and protect the data being transmitted.

We begin our application by including Boost.Asio library headers and headers of some components of standard C++ libraries that we will need to implement in our application:

[PRE48]

The `<boost/asio/ssl.hpp>` header contains types and functions providing integration with OpenSSL library.

Next, we define a class responsible for handling a single client by reading the request message, processing it, and then sending back the response message. This class represents a single service provided by the server application and is named correspondingly—`Service`:

[PRE49]

Next, we define another class that represents a high-level *acceptor* concept (as compared to the low-level acceptor represented by the `asio::ip::tcp::acceptor` class). This class is responsible for accepting connection requests arriving from clients and instantiating objects of the `Service` class, which will provide the service to connected clients. This class is called `Acceptor`:

[PRE50]

Now we define a class that represents the server itself. The class is named correspondingly—`Server`:

[PRE51]

Eventually, we implement the `main()` application entry point function that demonstrates how to use the `Server` class. This function is identical to the one defined in the recipe from [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, that we took as a base for this recipe:

[PRE52]

Note that the last two components of the server application, namely, the `Server` class and the `main()` application entry point function are identical to the corresponding components defined in the recipe from [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, that we took as a base for this recipe.

## How it works…

The sample server application consists of four components: the `Service`, `Acceptor`, and `Server` classes and the `main()`, application entry point function, which demonstrates how to use the `Server` class. Because the source code and the purpose of the `Server` class and the `main()` entry point function are identical to those of the corresponding components defined in the recipe from [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, that we took as a base for this recipe, we will not discuss them here. We will only consider the `Service` and `Acceptor` classes that were updated to provide support for SSL/TLS protocol.

### The Service class

The `Service` class is the key functional component in the application. While other components are infrastructural in their purpose, this class implements the actual function (or service) required by the clients.

The `Service` class is quite simple and consists of a single method `handle_client()`. As its input argument, this method accepts a reference to an object representing an SSL stream that wraps a TCP socket connected to a particular client.

The method begins with performing an SSL/TLS **handshake** by invoking the `handshake()` method on the `ssl_stream` object. This method is synchronous and does not return until the handshake completes or an error occurs.

After the handshake has completed, a request message is synchronously read from the SSL stream until a new line ASCII symbol `\n` is encountered. Then, the request is processed. In our sample application, request processing is trivial and dummy and consists in running a loop performing one million increment operations and then putting the thread to sleep for half a second. After this, the response message is prepared and sent back to the client.

Exceptions that may be thrown by the Boost.Asio functions and methods are caught and handled in the `handle_client()` method and are not propagated to the method's caller so that, if handling of one client fails, the server continues working.

Note that the `handle_client()` method is very similar to the corresponding method defined in the recipe *Implementing a synchronous iterative TCP server*, from [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, that we took as a base for this recipe. The difference consists in the fact that in this recipe, the `handle_client()` method operates on an object representing an SSL stream as opposed to an object representing a TCP socket being operated on in the base implementation of the method. Besides, an additional operation—an SSL/TLS handshake—is performed in the method defined in this recipe.

### The Acceptor class

The `Acceptor` class is a part of the server application infrastructure. Each object of this class owns an instance of the `asio::ssl::context` class named `m_ssl_context`. This member represents an **SSL context**. Basically, the `asio::ssl::contex` class is a wrapper around the `SSL_CTX` data structure defined by OpenSSL library. Objects of this class contain global settings and parameters used by other objects and functions involved in the process of communication using SSL/TLS protocol.

The `m_ssl_context` object, when instantiated, is passed a `asio::ssl::context::sslv23_server` value to its constructor to designate that the SSL context will be used by the application playing a role of a *server* only and that multiple secure protocols should be supported, including multiple versions of SSL and TLS. This value defined by Boost.Asio corresponds to a value representing a connection method returned by the `SSLv23_server_method()` function defined by OpenSSL library.

The SSL context is configured in the `Acceptor` class' constructor. The context options, password callback and files containing digital certificates, and private keys and Diffie-Hellman protocol parameters, are specified there.

After SSL context has been configured, the `listen()` method is called on the acceptor object in the `Acceptor` class' constructor to start listening for connection requests from the clients.

The `Acceptor` class exposes a single `accept()` public method. This method, when called, first instantiates an object of the `asio::ssl::stream<asio::ip::tcp::socket>` class named `ssl_stream`, representing an SSL/TLS communication channel with the underlying TCP socket. Then, the `accept()` method is called on the `m_acceptor` acceptor object to accept a connection. The TCP socket object owned by `ssl_stream`, returned by its `lowest_layer()` method, is passed to the `accept()` method as an input argument. When a new connection is established, an instance of the `Service` class is created and its `handle_client()` method is called, which performs communication with the client and request handling.

## See also

*   The *Implementing synchronous iterative TCP server* recipe from [Chapter 4](ch04.html "Chapter 4. Implementing Server Applications"), *Implementing Server Applications*, provides more information on how to implement a synchronous TCP server used as a base for this recipe.