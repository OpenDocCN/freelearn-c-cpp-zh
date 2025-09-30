# 示例程序

本书代码库位于[`github.com/codeplea/hands-on-network-programming-with-c`](https://github.com/codeplea/hands-on-network-programming-with-c)，包含 44 个示例程序。这些程序在本书中均有详细解释。

# 代码许可证

本书代码库中提供的示例程序均采用 MIT 许可证发布，许可证文本如下：

```cpp
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to
whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

```

# 本书包含的代码

以下是本书按章节划分的 44 个示例程序列表。

# 第一章 – 网络和协议简介

本章包含以下示例程序：

+   `win_init.c`: 初始化 Winsock 的示例代码（仅适用于 Windows）

+   `win_list.c`: 列出所有本地 IP 地址（仅适用于 Windows）

+   `unix_list.c`: 列出所有本地 IP 地址（仅适用于 Linux 和 macOS）

# 第二章 – 掌握套接字 API

本章包含以下示例程序：

+   `sock_init.c`: 包含所有必要的头文件并初始化的示例程序

+   `time_console.c`: 将当前日期和时间打印到控制台

+   `time_server.c`: 提供显示当前日期和时间的网页服务

+   `time_server_ipv6.c`: 与前面的代码相同，但监听 IPv6 连接

+   `time_server_dual.c`: 与前面的代码相同，但监听 IPv6/IPv4 双栈连接

# 第三章 – TCP 连接的深入概述

本章包含以下示例程序：

+   `tcp_client.c`: 建立 TCP 连接并从控制台发送/接收数据。

+   `tcp_serve_toupper.c`: 使用`select()`处理多个连接的 TCP 服务器。将接收到的数据全部转换为大写后回显给客户端。

+   `tcp_serve_toupper_fork.c`: 与前面的代码相同，但使用`fork()`代替`select()`。仅适用于 Linux 和 macOS。

+   `tcp_serve_chat.c`: 一个将接收到的数据转发给所有其他已连接客户端的 TCP 服务器。

# 第四章 – 建立 UDP 连接

本章包含以下示例程序：

+   `udp_client.c`: 从控制台发送/接收 UDP 数据

+   `udp_recvfrom.c`: 使用`recvfrom()`接收一个 UDP 数据报

+   `udp_sendto.c`: 使用`sendto()`发送一个 UDP 数据报

+   `udp_serve_toupper.c`: 监听 UDP 数据并将其全部转换为大写后回显给发送者

+   `udp_serve_toupper_simple.c`: 与前面的代码相同，但不使用`select()`

# 第五章 – 主机名解析和 DNS

本章包含以下示例程序：

+   `lookup.c`: 使用`getaddrinfo()`查找给定主机名的地址

+   `dns_query.c`: 编码并发送 UDP DNS 查询，然后监听并解码响应

# 第六章 – 构建简单的 Web 客户端

本章包含以下示例程序：

+   `web_get.c`: 一个从给定 URL 下载网络资源的最小化 HTTP 客户端

# 第七章 – 构建简单的 Web 服务器

本章包含以下示例程序：

+   `web_server.c`: 能够提供静态网站服务的最小化 Web 服务器

+   `web_server2.c`: 一个最小化 Web 服务器（无全局变量）

# 第八章 – 使你的程序发送电子邮件

本章包括以下示例程序：

+   `smtp_send.c`: 一个简单的发送电子邮件的程序

# 第九章 – 使用 HTTPS 和 OpenSSL 加载安全 Web 页面

本章的示例使用 OpenSSL。编译时请确保链接到 OpenSSL 库（`-lssl -lcrypto`）：

+   `openssl_version.c`: 一个报告已安装 OpenSSL 版本的程序

+   `https_simple.c`: 一个使用 HTTPS 请求网页的最小程序

+   `https_get.c`: 第六章的 HTTP 客户端，*构建简单的 Web 客户端*，修改为使用 HTTPS

+   `tls_client.c`: 第三章的*TCP 连接深入概述*中的 TCP 客户端程序，修改为使用 TLS

+   `tls_get_cert.c`: 从 TLS 服务器打印证书

# 第十章 – 实现安全 Web 服务器

本章的示例使用 OpenSSL。编译时请确保链接到 OpenSSL 库（`-lssl -lcrypto`）：

+   `tls_time_server.c`: 第二章的*掌握套接字 API*中的时间服务器，修改为使用 HTTPS

+   `https_server.c`: 第七章的*构建简单的 Web 服务器*中的 Web 服务器，修改为使用 HTTPS

# 第十一章 – 使用 libssh 建立 SSH 连接

本章的示例使用`libssh`。编译时请确保链接到`libssh`库（`-lssh`）：

+   `ssh_version.c`: 一个报告`libssh`版本的程序

+   `ssh_connect.c`: 一个建立 SSH 连接的最小客户端

+   `ssh_auth.c`: 一个尝试使用密码进行 SSH 客户端认证的客户端

+   `ssh_command.c`: 一个通过 SSH 执行单个远程命令的客户端

+   `ssh_download.c`: 一个通过 SSH/SCP 下载文件的客户端

# 第十二章 – 网络监控和安全

本章不包含任何示例程序。

# 第十三章 – 套接字编程技巧和陷阱

本章包括以下示例程序：

+   `connect_timeout.c`: 展示如何提前超时`connect()`调用。

+   `connect_blocking.c`: 用于与`connect_timeout.c`进行比较。

+   `server_reuse.c`: 展示`SO_REUSEADDR`的使用。

+   `server_noreuse.c`: 用于与`server_reuse.c`进行比较。

+   `server_crash.c`: 这个服务器在客户端断开连接后故意写入 TCP 套接字。

+   `error_text.c`: 展示如何获取错误代码描述。

+   `big_send.c:` 连接后发送大量数据的 TCP 客户端。用于展示`send()`的阻塞行为。

+   `server_ignore.c`: 一个接受连接然后简单地忽略它们的 TCP 服务器。用于展示`send()`的阻塞行为。

+   `setsize.c`: 展示`select()`可以处理的套接字最大数量。

# 第十四章 – 物联网的 Web 编程

本章不包含任何示例程序。
