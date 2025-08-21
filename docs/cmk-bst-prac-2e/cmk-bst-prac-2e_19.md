# 附录：为 CMake 做贡献与进一步阅读资料

这是一个漫长的旅程，我们已经学到了很多关于 CMake 的知识。然而，正如你现在可能已经意识到的，CMake 是一个庞大的生态系统，一本书并不足以涵盖所有可以讨论的主题。因此，在本章中，我们将看一下那些能帮助你更好理解 CMake 的资源，以及参与 CMake 项目的方式。

CMake 是一个灵活的工具，广泛应用于软件行业中的许多项目。因此，CMake 拥有一个不断壮大的支持社区。网上有大量资源可供学习和解决你可能遇到的 CMake 问题。

为了理解本章中分享的技能，我们将涵盖以下主要内容：

+   寻找 CMake 社区的途径

+   为 CMake 做贡献

+   推荐的书籍和博客

让我们开始吧！

# 前提条件

这是一个通读章节，没有实践或示例。所以，唯一的要求就是一台兼容的设备、一个安静的地方，当然，还有你的时间。

# 寻找 CMake 社区的途径

在深入 CMake 之后，你可能会有与他人交流想法的需求，或者寻找一个平台，向可能知道答案的人提问。为此，我为你提供了一些在线平台的推荐。

## Stack Overflow

**Stack Overflow** 是一个受欢迎的问答平台，也是大多数开发者的首选。如果你遇到 CMake 问题或有任何疑问，可以先在 Stack Overflow 上搜索问题的答案。很有可能有人遇到过相同的问题，或者以前问过类似的问题。你还可以查看热门问题列表，发现一些使用 CMake 的新方法。

提问时，确保给你的问题加上 `cmake` 标签。这样，感兴趣回答 CMake 相关问题的人就能更容易找到你的问题。你可以访问 Stack Overflow 的主页：[`stackoverflow.com/`](https://stackoverflow.com/)。

## Reddit (r/cmake)

`r/cmake` 子版块，其中包含 CMake 相关的问题、公告和分享。你可以发现许多有用的 GitHub 仓库，获取 CMake 最新版本的通知，发现博客文章和资料，帮助你解决问题。你可以访问 `r/cmake` 子版块：[`www.reddit.com/r/cmake/`](https://www.reddit.com/r/cmake/)。

## CMake 讨论论坛

**CMake 讨论论坛** 是 CMake 开发者和用户交流的主要平台。它完全专注于 CMake 相关的内容。论坛包含公告、如何使用 CMake 的指南、社区空间、CMake 开发空间，以及许多你可能感兴趣的其他内容。你可以访问该论坛：[`discourse.cmake.org/`](https://discourse.cmake.org/)。

## Kitware CMake GitLab 仓库

Kitware 的 CMake 仓库也是一个很好的资源，可以帮助你解决可能遇到的问题。尝试在[`gitlab.kitware.com/cmake/cmake/-/issues`](https://gitlab.kitware.com/cmake/cmake/-/issues)上搜索你遇到的问题。很有可能其他人已经报告了类似的问题。如果没有，你可以遵循 CMake 的贡献规则创建一个新的问题。

上述列表并不全面，网上还有许多其他论坛。以下四个平台已经足够让你入门。接下来，我们将讨论如何为 CMake 项目本身做贡献。

# 为 CMake 做贡献

如你所知，CMake 是由 Kitware 开发的开源软件。Kitware 在[`gitlab.kitware.com/cmake`](https://gitlab.kitware.com/cmake)的专用 GitLab 实例中维护 CMake 的开发活动。所有内容都以开源且透明的形式提供，意味着参与 CMake 的开发相对容易。你可以查看问题、合并请求，并参与 CMake 的开发。如果你认为你发现了 CMake 中的 bug，或者想提出功能请求，可以在[`gitlab.kitware.com/cmake/cmake/-/issues`](https://gitlab.kitware.com/cmake/cmake/-/issues)上创建一个新问题。如果你有改进 CMake 的想法，可以先通过创建一个问题来讨论这个想法。你还可以查看[`gitlab.kitware.com/cmake/cmake/-/merge_requests`](https://gitlab.kitware.com/cmake/cmake/-/merge_requests)上的开放合并请求，帮助审查正在开发的代码。

为开源软件做贡献对开源世界的可持续发展至关重要。请不要犹豫，以任何方便的方式帮助开源社区。你提供的帮助可能很小，但小小的贡献会迅速积累成更大的成就。接下来，我们将查看一些你可能会觉得有用的阅读材料。

# 推荐的书籍和博客

关于 CMake 有许多书籍、博客和资源。以下是一些精选的你可能会觉得有用的资源列表。这些资源将帮助你进一步了解 CMake，拓宽你的视野：

+   **CMake 官方** **文档**：[`cmake.org/documentation/`](https://cmake.org/documentation/)。

    +   CMake 的官方文档。内容非常全面并且是最新的。

+   *专业 CMake：实用* *指南*：[`crascit.com/professional-cmake/`](https://crascit.com/professional-cmake/)。

    这是一本由 CMake 的共同维护者 Craig Scott 编写的全面书籍。它非常详细，包含许多在其他地方找不到的信息。如果你想深入理解 CMake，强烈推荐阅读这本书。

+   `awesome-cmake`：[`github.com/onqtam/awesome-cmake`](https://github.com/onqtam/awesome-cmake)。

    一个关于 CMake 的庞大资源集合。内容非常广泛并且定期更新。

+   *开始使用 CMake：有用的* *资源*：[`embeddedartistry.com/blog/2017/10/04/getting-started-with-cmake-helpful-resources/`](https://embeddedartistry.com/blog/2017/10/04/getting-started-with-cmake-helpful-resources/)。

    由 Embedded Artistry 收集的有用 CMake 资源整理。

+   *现代* *CMake 简介*：[`cliutils.gitlab.io/modern-cmake/`](https://cliutils.gitlab.io/modern-cmake/)。

    一本在线书籍，详细介绍了学习现代 CMake 的好资源。它由 Henry Schreiner 和许多其他贡献者编写。

+   *更现代的* *CMake*：[`hsf-training.github.io/hsf-training-cmake-webpage/01-intro/index.html`](https://hsf-training.github.io/hsf-training-cmake-webpage/01-intro/index.html)。

    这是一本跟随 *现代 CMake 简介* 的书，由 HEP 软件基金会编写。

+   *更现代的 CMake - Deniz Bahadir - Meeting C++* *2018*：[`www.youtube.com/watch?v=y7ndUhdQuU8`](https://www.youtube.com/watch?v=y7ndUhdQuU8)。

    这段 YouTube 视频是 Deniz Bahadir 在 Meeting C++ 2018 上的演讲。其主要目的是提供关于如何正确使用 CMake 的技巧。

+   *深入 CMake 面向库作者 - Craig Scott - CppCon* *2019*：[`www.youtube.com/watch?v=m0DwB4OvDXk`](https://www.youtube.com/watch?v=m0DwB4OvDXk)。

    这段 YouTube 视频是由 CMake 项目的共同维护者 Craig Scott 在 CppCon 上的演讲，内容涉及面向库开发的 CMake 主题。

+   *C++Now 2017：Daniel Pfeifer “有效的* *CMake”*：[`www.youtube.com/watch?v=bsXLMQ6WgIk`](https://www.youtube.com/watch?v=bsXLMQ6WgIk)。

    这段 YouTube 视频是 Daniel Pfeifer 关于有效使用 CMake 的演讲，涵盖了 CMake 的整体使用。

+   *CMake init – CMake 项目* *初始化工具*：[`github.com/friendlyanon/cmake-init`](https://github.com/friendlyanon/cmake-init)。

    一个强大的脚本，用于从头开始初始化各种用途的 CMake 项目。该仓库还包含许多不同种类 CMake 项目的示例链接。

话虽如此，我们已经到达了另一个章节的结尾。接下来，我们将总结本章所学内容。

# 摘要

在本章中，我们简要讨论了你可以在网上找到的 CMake 社区、贡献 CMake 以及一些很好的阅读和观看推荐。关于 CMake 的材料和演讲数量庞大，内容也在日益增长。时刻关注 CMake 的更新，并定期访问你选择的论坛，保持信息的同步。

话虽如此，如果你已经来到这里并阅读这段文字，那么恭喜你！你已经完成了我们在本书中希望涵盖的所有主题。这是最后一章内容。不要忘记将你从本书中学到的知识应用并实践到日常工作流中。我们很享受一起走过的这段旅程，希望你从本书中获得的知识能够为你带来帮助。
