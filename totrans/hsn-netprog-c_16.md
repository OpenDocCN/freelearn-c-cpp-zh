# Network Monitoring and Security

In this chapter, we will look at common tools and techniques for **network monitoring**. These techniques can be useful to both alert us to developing problems, and to help troubleshoot existing problems. Network monitoring can be important from a network security standpoint, where it may be useful to detect, log, or even prevent network intrusions.

The following topics are covered in this chapter:

*   Checking host reachability
*   Displaying a connection route
*   Showing open ports
*   Listing open connections
*   Packet sniffing
*   Firewalls and packet filtering

# Technical requirements

This chapter contains no C code. Instead, it focuses on useful tools and utilities.

The utilities and tools used in this chapter are either built in to your operating system or are available as free, open source software. We'll provide instructions for each tool as it is introduced.

# The purpose of network monitoring

Network monitoring is a common IT term that has a broad implication. Network monitoring can refer to the practice, techniques, and tools used to provide insight into the status of a network. These techniques are used to monitor the availability and performance of networked systems and troubleshoot problems.

Some reasons you may want to practice network monitoring include the following:

*   To detect the reachability of networked systems
*   To measure the availability of networked systems
*   To determine the performance of networked systems
*   To inform decisions about network resource allocation
*   To aid in troubleshooting
*   To benchmark performance
*   To reverse engineer a protocol
*   To debug a program

In this chapter, we look at a small subset of conventional network monitoring techniques that may be useful when implementing networked programs.

When developing or deploying networked programs, it is often the case that you run into problems. When this happens, you are faced with two possibilities. The first possibility is that your program could have a bug in it. The second possibility is that your problem is caused by a network issue. The methods covered in this chapter help to identify and troubleshoot network problems.

One of the most fundamental questions you could ask is, can this system reach that system? The **Ping** utility, which is perhaps the most elementary network tool, is designed to answer just that question. Let's consider its usage next.

# Testing reachability

Perhaps the most basic network monitoring tool is Ping. Ping uses the **Internet Control Message Protocol **(**ICMP**) to check whether a host is reachable. It also commonly reports the total round-trip time (latency). Ping is available as a built-in command or utility for all common operating systems.

The ICMP defines a set of special IP messages that are typically useful for diagnostic and control purposes. Ping works by using two of these messages: **echo request** and **echo reply**. The Ping utility sends an echo request ICMP message to a destination host. When that host receives the echo request, it should respond with an echo reply message.

When the echo reply is received, Ping knows that the destination host is reachable. Ping can also report the round-trip time from when the echo request was sent to when the echo reply was received. ICMP echo messages are usually small and easy to process, so this round-trip time often serves as a best-case estimate of network latency.

The Ping utility takes one argument: the hostname or address you're requesting a response from. Ping on Unix-based systems will send packets continuously. On Windows, pass in the `-t` flag for this behavior. Press *Ctrl +C* to stop.

The following screenshot shows using the Ping utility on `example.com`:

![](img/55e7638f-2e39-4d66-b4da-d49420c7b877.png)

In the preceding screenshot, you can see that several ping messages were sent, and each received a response. The round-trip time is reported for each ping message, and a summary of all sent packets is also reported. The summary includes the minimum, average, and maximum round-trip message time.

It is also possible to use ping to send larger messages. On Linux and macOS, the `-s` flag specifies packet size. On Windows, the `-l` flag is used. It is sometimes interesting to see how packet size affects latency and reliability.

The following screenshot shows pinging `example.com` with a larger 1,000-byte ping message:

![](img/a0e42745-2d51-47f7-a959-2c7e76163b4c.png)

If ping fails to receive an echo reply, it does not necessarily mean that the target host is unreachable. It only means that ping did not receive the expected reply. This could be because the target machine ignored the echo request. However, most systems *are* set up to respond to ping requests properly, and an echo request timeout usually means the target system is unreachable.

Sometimes, just knowing that a host is reachable is enough, but sometimes you'll want more information. It can be useful to know the exact path an IP packet takes through the network. The **traceroute** utility provides this information.

# Checking a route

Although we briefly introduced traceroute in [Chapter 1](e3e07fa7-ff23-4871-b897-c0d4551e6422.xhtml), *Introducing Networks and Protocols*, it is worth revisiting in more detail.

While Ping can tell us whether a network path exists between two systems, traceroute can reveal what this path actually is.

Traceroute is used with one argument: the hostname or address you want to map a route to.

On Windows, traceroute is called **tracert**. Tracert works in a very similar way to the traceroute utility found on Linux and macOS.

The following screenshot shows the traceroute utility printing the routers used to deliver data to `example.com`. The `-n` flag tells `traceroute` not to perform reverse-DNS lookups for each hop. These lookups are rarely useful and omitting them saves a bit of time and screen space:

![](img/67a51c03-0005-4370-8e1a-c606550380f5.png)

The preceding screenshot shows that there are four or five routers (or hops) between us and the destination system at `example.com`. Traceroute also shows the round-trip time to each intermediate router.

Traceroute sends three messages to each router. This often exposes multiple network paths, and there is no guarantee that any two messages will take precisely the same path.

In the preceding example, we see that the message must first pass through `23.92.28.3`. From there, it goes to one of three different systems, which are listed. The message continues until it reaches the destination at hop five or six, depending on the exact path it takes through the network.

This illustrates an interesting point: you shouldn't assume that two consecutive packets take the same network path.

# How traceroute works

To understand how traceroute works, we must understand a detail of the **Internet Protocol** (**IP**). Each IP packet header contains a field called **Time to Live** (**TTL**). TTL is the maximum number of seconds that a packet should live on the network before being discarded. This is important to keep an IP packet from simply persisting (which is, going in an endless loop) over the network.

TTL time intervals under one second are rounded up. This means that, in practice, each router that handles an IP packet decrements the TTL field by `1`. Therefore, TTL is often used as a hop-count. That is to say that the TTL field simply represents the number of hops a packet can still take over the network.

The traceroute utility uses TTL to identify intermediate routers in a network. Traceroute begins by addressing a message (for example, a UDP datagram or an ICMP echo request) to the destination host. However, traceroute sets the TTL field to a value of `1`. When the very first router in the connection path receives this message, it decrements TTL to zero. The router then realizes that the message has expired and discards it. A well-behaved router then sends an ICMP **Time Exceeded** message back to the original sender. Traceroute uses this **Time Exceeded** message to identify the first router in the connection.

Traceroute repeats this same process with additional messages. The second message is sent using a TTL of `2`, and that message identifies the second hop in the network path. The third message is sent using a TTL of `3`, and so on. Eventually, the message reaches its final destination and traceroute has mapped the entire network path.

The following diagram illustrates the method used by traceroute:

![](img/8b3283b5-c84d-4807-b5e5-627730ea76f1.png)

In the preceding diagram, the first message is sent with a TTL of 1\. Router 1 doesn't forward this message, but instead returns an ICMP **Time Exceeded** message. The second message is sent with a TTL of 2\. It makes it to the second router before timing out. The third message makes it to the destination, which replies with an **Echo Reply** message. (If this traceroute were UDP-based, it would expect to receive an ICMP **Port Unreachable** message instead.)

Not all routers return an ICMP **Time Exceeded** message, and some networks filter out these messages. In these cases, traceroute will have no way to know these routers' addresses. Traceroute prints an asterisk instead. Theoretically, if a router exists in the connection path that doesn't decrement the TTL field, then traceroute has no way of knowing that this router exists.

Now that we've covered the way ping and traceroute work, you may be wondering how they could be implemented in C code. Unfortunately, despite their simple algorithms, this is easier said than done. Keep reading to learn more.

# Raw sockets

You may be interested in implementing your own network testing tools. The ping tool seems pretty simple, and it is. Unfortunately, the socket programming APIs we've been working with do not provide access at the IP level that the ICMP is built on.

The socket programming API does provide access to **raw sockets**, in theory. With raw sockets, a C program can construct the exact IP packet to send. That is, a C programmer could construct an ICMP packet from scratch and send it over the network. Raw sockets also allow for programs to receive uninterpreted packets from the network directly. In this case, the user program would be responsible for deconstructing and interpreting the ICMP packet, not the operating system.

On systems with raw socket support, getting started can be as simple as changing your `socket()` function invocation to the following:

```cpp
socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
```

However, the problem is that raw sockets aren't universally supported. It is a difficult subject to approach in a cross-platform way. Windows, in particular, has varying support for raw sockets depending on the OS version. Recent versions of Windows have virtually no support for raw sockets. For this reason, we won't cover raw sockets in any more detail here.

Now that we've covered two basic tools for network troubleshooting, let's next look at tools that inform us about our own system's relationship to the network.

# Checking local connections

It is often useful to know what connections are being made on your local machine. The `netstat` command can help with that. **Netstat** is available on Linux, macOS, and Windows. Each version differs a little in the command-line options and output, but the general usage principles are the same.

I recommend running `netstat` with the `-n` flag. This flag prevents `netstat` from doing reverse-DNS lookups on each address and has the effect of speeding it up significantly.

On Linux, we can use the following command to show open TCP connections:

```cpp
netstat -nt
```

The following screenshot shows the result of running this command on Linux:

![](img/811160a7-9fda-404b-800c-113f9b13dd74.png)

In the preceding screenshot, you can see that `netstat` shows six columns. These columns display the protocol, the sending and receiving queue, the local address, the foreign address, and the connection state. In this example, we see that there are three connections to port `80`. It is likely that this computer is loading up three web pages (as HTTP uses port `80`).

On Windows, the `netstat -n -p TCP` command shows the same information, except it omits the socket queue information.

The queue information, displayed on Unix-based systems, represents how many bytes the kernel has queued up waiting for the program to read, or the number of bytes sent but not acknowledged by the foreign host. Small numbers are healthy, but if these numbers became large, it could indicate a problem with the network or a bug in the program.

It is also useful to see which program is responsible for each connection. Use the `-p` flag on Unix-based systems for this. On Windows, the `-o` flag shows PID and the `-b` flag shows executable names.

If you're working on a server, it is often useful to see which listening sockets are open. The `-l` flag instructs Unix-based `netstat` to show only listening sockets. The following screenshot shows listening TCP server sockets, including the program name:

![](img/2e60aa53-efa0-4a2a-bd87-08a2826f856c.png)

In the preceding screenshot, we can see that this system is running a DNS resolver (`systemd-resolve` on port `53`, IPv4 only), an SSH daemon (`sshd` on port `22`), a printer service (`cupsd` on port `631`), and a web server (`apache2` on port `80`, IPv6 only).

On Windows, `netstat` doesn't have an easy way to display only listening sockets. Instead, you can use the `-a` flag to display everything. Filtering out only listening TCP sockets can be accomplished with the following command:

```cpp
netstat -nao -p TCP | findstr LISTEN
```

The following screenshot shows using `netstat` on Windows to show only listening TCP sockets:

![](img/3b0bcc65-a611-4de6-a19c-c2bdbc4314be.png)

With an idea of what programs are communicating from our machine, it may also be useful to see what they are communicating. Tools such as `tcpdump` and `tshark` specialize in this, and we will cover them next.

# Snooping on connections

In addition to seeing which sockets are open on our computer, we can also capture the exact data being sent and received.

We have a few tooling options for this:

*   **tcpdump** is a commonly used program for packet capture on Unix-based systems. It is not available on modern Windows systems, however.
*   **Wireshark** is a very popular network protocol analyzer that includes a very nice GUI. Wireshark is free software, released under the GNU GPL license, and available on many platforms (including Windows, Linux, and macOS).

Included with Wireshark is Tshark, a command-line-based tool that allows us to dump and analyze network traffic. Programmers often prefer command-line tools for their simple interfaces and ease of scripting. They have the additional benefit on being usable on systems where GUIs may not be available. For these reasons, we focus on using Tshark in this section.

Tshark can be obtained from [https://www.wireshark.org](https://www.wireshark.org).

If you're running Linux, it is likely that your distro provides a package for Tshark. For example, on Ubuntu Linux, the following commands will install Tshark:

```cpp
sudo apt-get update
sudo apt-get install tshark
```

Once installed, Tshark is very easy to use.

You first need to decide which network interface(s) you want to use to capture traffic. The desired interface or interfaces are passed to `tshark` with the `-i` flag. On Linux, you can listen to all interfaces by passing `-i any`. However, Windows doesn't provide the `any` interface. To listen on multiple interfaces with Windows, you need to enumerate them separately, for example, `-i 1 -i 2 -i 3`.

Tshark lists the available interfaces with the `-D` flag. The following screenshot shows Tshark on Windows enumerating the available network interfaces:

![](img/22846afd-df30-4644-8a21-7b7d3047f7b8.png)

If you want to monitor local traffic (that is., where the communication is between two programs on the same computer), you will want to use the `Loopback` adapter.

Once you've identified the network interface you would like to monitor, you can start Tshark with the `-i` flag and begin capturing traffic. Use *Ctrl* + *C* to stop capturing. The following screenshot shows Tshark in use:

![](img/bac22f5b-e23a-49ce-9a1a-430a09f497e4.png)

The preceding screenshot represents only a few seconds of running Tshark on a typical Windows desktop. As you can see, there is a lot of traffic into and out of an even relatively idle system.

To cut down on the noise, we need to use a capture filter. Tshark implements a small language that allows easy specification of which packets to capture and which to ignore.

Explaining filters may be easiest by way of example.

For example, if we want to capture only traffic to or from the IP address `8.8.8.8`, we will use the `host 8.8.8.8` filter.

In the following screenshot, we've run Tshark with the `host 8.8.8.8` filter:

![](img/b8db829c-454e-44f3-8074-a4d9971cd674.png)

You can see that while Tshark was running, it captured two packets. The first packet is a DNS request sent to `8.8.8.8`. Tshark informs us that this DNS request is for the A record of `example.com`. The second packet is the DNS response received from `8.8.8.8`. Tshark shows that the DNS query response indicators that the A record for `example.com` is `93.184.216.34`.

Tshark filters also support the Boolean operators `and`, `or`, and `not`. For example, to capture only traffic involving the IP addresses `8.8.8.8` and `8.8.4.4` you can use the `host 8.8.8.8` filter or the `host 8.8.4.4` filter.

Filtering by port number is also very useful and can be done with `port`. For example, the following screenshot shows Tshark being used to capture traffic to `93.184.216.34` on port `80`:

![](img/e212f568-aa24-40a2-be4b-220bae29dcf0.png)

In the preceding screenshot, we see that Tshark was run with the `tshark -i 5 host 93.184.216.34 and port 80` command. This has the effect of capturing all traffic on network interface `5` that is to or from `93.184.216.34` port `80`.

TCP connections are sent as a series of packets. Although Tshark reports capturing 11 packets, these are all associated with only one TCP connection.

So far, we've been using Tshark in a way that causes it to display a summary of each packet. Often this is enough, but sometimes you will want to be able to see the entire contents of a packet.

# Deep packet inspection

If we pass `tshark` the `-x` flag, it displays an ASCII and hex dump of every packet captured. The following screenshot shows this usage:

![](img/7f999317-cb19-4d34-9cd1-daeeba517952.png)

In the preceding screenshot, you can see that entire IP packets are being dumped. In this case, we see the first three packets represent a new TCP connection's three-way handshake. The fourth packet contains an HTTP request.

This direct insight into each packet's contents isn't always convenient. It's often more practical to capture packets to a file and do the analysis later. The `-w` option is used with `tshark` to capture packets to a file. You may also want to use the `-c` option to limit the number of packets captured. This simple precaution protects against accidentally filling your entire hard drive with network traffic.

The following screenshot shows using Tshark to capture 50 packets to a file named `capture.pcap`:

![](img/b616fc51-0e9a-47c6-b223-7df752700d39.png)

Once the traffic is written to a file, we can use Tshark to analyze it at our leisure. Simply run `tshark -r capture.pcap` to get started. For text-based protocols (such as HTTP or SMTP), it is also often useful to open the capture file in a text editor for analysis.

The ultimate way to analyze captured traffic is with **Wireshark**. Wireshark allows you to load a `capture` file produced with `tshark` or `tcpdump` and analyze it with a very nice GUI. Wireshark is also able to understand many standard protocols.

The following screenshot shows using Wireshark to display the traffic captured from Tshark:

![](img/5e9becfa-3a7f-4954-a686-6a8d861a5bfc.png)

If the system you need to capture traffic on has a GUI, you can also use Wireshark to capture your traffic directly.

If you play around with `tshark` and Wireshark, you will quickly see that it is easy to inspect network protocols at a deep level. You may even find some questionable security choices made by software running on your own system.

Although we've been focusing on monitoring only network traffic on our local system, it is also possible to monitor all local network traffic.

# Capturing all network traffic

`tshark` is only able to see traffic that arrives on your machine. This usually means you can only use it to monitor traffic from applications on your computer.

To capture all internet traffic on your network, you must somehow arrange for that traffic to arrive at your system, even though it usually wouldn't. There are two basic methods to do this.

The first method is by using a router that supports mirroring. This feature makes it mirror all traffic to a given Ethernet port. If you are using such a router, you can configure it to mirror all network traffic to a particular port, and then plug your computer into that port. From that point, any traffic capturing tool, such as `tcpdump` or `tshark`, can be used to record this traffic.

The second way to sniff all internet traffic is to install a hub (or a switch with port mirroring) between your router and internet modem. Hubs work by mirroring all traffic to all ports. Hubs used to be a common way to build a network. However, they have mostly been replaced by more efficient switches.

With a hub installed between your router and Internet modem, you can then connect your system directly to this hub. With that setup, you can receive all the internet traffic into or out of the network.

To monitor all network traffic (such as, even traffic between devices on the same network), you need to build your network with either a hub or a switch that supports port mirroring.

It should also be noted, that with the right modem, you can potentially capture all Wi-Fi traffic wirelessly.

From a security perspective, let this be instructive. You should never consider any network traffic to be secret. Only traffic which is secured appropriately using encryption is resistant to monitoring.

Now that we've shown several tools and techniques for testing and monitoring network traffic, let's consider the important topic of network security next.

# Network security

**Network security** encompasses the tools, techniques, and practices to protect a network from threats. These tools include both hardware and software and can protect against a variety of threats.

Although the topic is too broad for much detail here, we will cover a few topics that you are likely to encounter.

**Firewalls** are one of the most common network security techniques. Firewalls act as a barrier between one network and another. As commonly used, they monitor network traffic and allow or block traffic based on a defined set of rules.

Firewalls come in two types: software and hardware. Most operating systems provide software firewalls now. Software firewalls are typically configured to deny incoming connections unless a rule is set up to allow it explicitly.

It is also possible to configure software firewalls to deny outgoing traffic by default. In this case, programs aren't allowed to establish new connections unless a specific rule is added to the firewall configuration first.

Do be careful about assuming that firewalls catch all traffic. For example, the Windows 10 firewall can be configured to deny all outgoing traffic by default, but it still lets DNS requests through. An attacker could take advantage of this by using DNS requests to exfiltrate data even though the user thinks that they are protected via the Windows firewall.

Hardware firewalls come with various capabilities. Often, they are configured to block any incoming connections that don't match predefined rules. On networks without firewalls, routers often implicitly serve this same purpose. If your router provides network address translation, then it wouldn't even know what to do with an incoming connection unless a port forwarding rule has been pre-established for it.

Although it is important to understand the basics of network security on a wide level, as C programmers, we are often more concerned with the security of our own programs. Security is not something that the C language makes easy, so let's consider application-level security in more detail now.

# Application security and safety

When programming in C, special consideration must be given to security. This is because C is a low-level programming language that gives direct access to system resources. Memory management, for example, must be done manually in C, and a mistake in memory management could allow a network attacker to write and execute arbitrary code.

With C, it is vital to ensure that allocated memory buffers aren't written past the end. That is, whenever you copy data from the network into memory, you must ensure that enough memory was allocated to store the data. If your program misses this even once, it potentially allows a window for an attacker to gain control of your program.

Memory management, from a security perspective, isn't a concern with many higher-level programming languages. In many programming languages, it isn't even possible to write outside of allocated memory. Of course, these languages also can't provide the precise control of memory layout and data structures that C programmers enjoy.

Even if you are careful to manage memory perfectly, there are still many more security gotchas to be on the lookout for. When implementing any network protocol, you should never assume that the data your program receives will adhere to the protocol specification. If your program does make these assumptions, it will be open to attack by a rogue program that doesn't. These protocol bugs are a concern in any programming language, not just C. Again, though, the thing about C is that these protocol implementation errors can quickly lead to memory errors, and memory errors quickly become serious.

If you're implementing a C program intended to run as a server, you should employ a **defense in depth** approach. That is, you should set up your program in such a manner that multiple defenses must be overcome before an attacker could cause damage.

The first layer of defense is writing your program without bugs. Whenever dealing with received network data, carefully consider what would happen if the received data was not at all what you're expecting. Make sure your program does the appropriate thing.

Also, don't run your program with any more privileges than needed for its functionality. If your program doesn't need access to a sensitive set of files, make sure your operating system doesn't allow it to access those files. Never run server software as root.

If you're implementing an HTTP or HTTPS server, consider not connecting your program directly to the internet. Instead, use a reverse proxy server as the first point of contact to the internet, and have your software interface only with the proxy server. This provides an additional layer of isolation against attacks.

Finally, consider not writing networked code in C at all, if you can find an alternative. Many C servers can be rewritten as CGI programs that don't directly interact with the network at all. TCP or UDP servers can often be rewritten to use `inetd`, and thus avoid the socket programming interface altogether. If your program needs to load web pages, consider using a well-tested library, such as `libcurl`, for that purpose instead of rolling your own.

We've covered a few network testing techniques so far. When deploying these techniques on live networks, it is important to be considerate. Let's finish with a note on etiquette.

# Network-testing etiquette

When network testing, it is always important to behave responsibly and ethically. Generally speaking, don't test someone else's network without their explicit permission. Doing otherwise could cause embarrassment at best, and land you in serious legal trouble at worst.

You should also be aware that some network testing techniques can set off alarms. For example, many network administrators monitor their network's load and performance characteristics. If you decide to load-test these networks without notice, you might set off automated alarms causing inconvenience.

Some other testing techniques can look like attacks. Port scanning, for example, is a useful technique where a tester tries establishing many connections on different ports. It's used to discover which ports on a system are open. However, it is a common technique used by malicious attackers to find weaknesses. Some system administrators consider port scans to be an attack, and you should never port scan a system without permission.

# Summary

In this chapter, we covered a lot of ground over the broad topics of network monitoring and security. We looked at tools useful for testing the reachability of networked devices. We learned how to trace a path through the network, and how to monitor connections made on our local machine. We also discovered how to log and inspect network traffic.

We discussed network security and how it may impact the C developer. By showing how network traffic can be directly inspected, we learned first-hand the importance of encryption for communication privacy. The importance of security at the application level was also discussed.

In the next chapter, we take a closer look at how our coding practices affect program behavior. We also discuss many essential odds and ends for writing robust network applications in C.

# Questions

Try these questions to test your knowledge from this chapter:

1.  Which tool would you use to test the reachability of a target system?
2.  Which tool lists the routers to a destination system?
3.  What are raw sockets used for?
4.  Which tools list the open TCP sockets on your system?
5.  What is one of the biggest concerns with security for networked C programs?

The answers to these questions can be found in [Appendix A](bd8b8f52-52cb-4d34-b01b-e907564bfece.xhtml), *Answers to Questions*.

# Further reading

For more information about the topics covered in this chapter, please refer to the following:

*   **RFC 792**: *Internet Control Message Protocol* ([https://tools.ietf.org/html/rfc792](https://tools.ietf.org/html/rfc792))
*   Wireshark ([https://www.wireshark.org](https://www.wireshark.org))