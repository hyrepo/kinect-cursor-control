# 简介
此程序为利用Kinect实现用手指隔空控制鼠标，是我另一个项目的一部分，因为在另外那个项目中鼠标的click是通过一种特殊的方式实现的，因此这个程序**只实现了用手控制鼠标的移动,并没有点击的功能**。相比Leapmotion,利用Kinect来控制鼠标可以大幅增加操控范围，使用者可以随意走动，而不是被固定在桌面前。

其中`双手控制`的操作窗口是实时更新的，操作更灵活，但是稳定性差。`单手控制`的操作窗口是固定的，稳定性好，不过只实现了右手。

# 运行环境
- Kinect for Windows V2
- Kinect SDK 2.0
- OpenCV 3.0

# 使用方法
首先需要让Kinect识别出你，建议距离0.5m以上4m以内并且正对摄像头，如果位置合适的话瞬间就能识别出，如果几秒钟都没有识别出来就请调节位置。然后将手放入操作窗口就可以控制鼠标了，绿色的点代表指尖。

# 效果
![](http://images.cnblogs.com/cnblogs_com/xz816111/786501/o_QQ%e5%9b%be%e7%89%8720160424152646.png)

# 原理介绍

### 指尖识别基础

程序的核心在于指尖识别，我发现网上关于指尖识别的资料不多，所以自己想了个很简单的办法，不过效果还不错。首先应该知道三点:

    1.  Kinect可以分辨出画面中哪一部分是人体
    2.  Kinect带有深度摄像头,可以获取物体到摄像头的距离
    3.  Kinect可以获取到人体的最多25个关节点的位置
    
### 减小搜索空间

为了寻找指尖，没有必要在整副画面中去搜索，那样会导致效率非常低。指尖肯定是在手部关节点(`Hand`)附近的，因此只需要获取到手部关节点的位置然后再拓展出一片区域，在这片区域里进行搜索就行了。之所以取`Hand`而不是取`HandTip`，是因为后者的稳定性非常差，即使在一个合适的距离正对Kinect也不一定能识别出来。
    
### 指尖识别

这时候就可以根据指尖的特征来进行指尖识别了。**不过实际上我识别出的不是指尖，而是指尖上方的一点。**此点的特征如下：

1. 不属于人体
2. 或者属于人体,但是和手部关节点不在同一个平面上(允许有误差)
3. 到手部关节点的位置在某个合适的范围内
4. 下面连续N(这里我取N为5)个像素都属于人体

首先，对于第一点很好理解，因为上面说了识别的不是指尖，而是指尖上方的一点。

第二点是用来处理手移动到身边正前方的时候的情况，比如手在胸前，这时候指尖上方的点都是属于人体的，不满足第一点。这里的误差指的是手和小臂成90度时的深度差，一般15cm左右。

第三点是为了消除两根手指根部之间的那个位置形成的误判，同时也进一步减小了搜索空间，正常情况下手指到手腕的距离都在10~25cm范围内，这里把拇指筛掉了，一般也不会用拇指去操作。如果要恢复拇指的话调整下参数就可以。

第四个条件筛选出了离手指尖最近的那个点。

### 确定操作窗口

为了便于操作和观察，我设置了一个操作窗口，位于肩部的左上方和右上方，根据操作手的左右而调整。这个窗口就代表着电脑的屏幕，手指在窗口里的位置就是鼠标在电脑屏幕上的位置。这里窗口的大小是根据关节点`Head`到`Neck`的距离作为单位长度算出来的，也就是说能根据人体到Kinect的距离来调整操作窗口的大小。同时这个窗口是实时更新的，会根据人体的位置而进行调整。

这里要说一下，如果操作的位置相对固定，那么建议识别出窗口后就不要再更新，将操作窗口固定，因为这样能够大幅度提高鼠标的稳定性，同时上传了一份以这种方式来做的代码，不过这份代码只实现了单手控制。

### 抖动消除

这是个不太好处理的问题，因为容易影响到正常操作。这里我设置了一个移动的阈值，如果和上一个位置相比，鼠标的位置改变很微小，那么就保持上次的位置不变。还可以再加入一个判断位置突变的阈值，如果当前位置和上一次位置相距太远，就可以判断为非法而筛去。

### 指尖位置与鼠标位置的转换

![](http://images.cnblogs.com/cnblogs_com/xz816111/786501/o_fsadfwefwe.PNG)
    
黑色框为程序里确定的操作窗口，大写的X和Y代表的是屏幕的宽和高，红色框为电脑屏幕，假设人的手指在![](http://images.cnblogs.com/cnblogs_com/xz816111/786501/o_x1y1.gif)的位置，如果想将鼠标也映射到同样的位置，那么就有 ![](http://images.cnblogs.com/cnblogs_com/xz816111/786501/o_tbgxfs.gif) 的等比关系成立。电脑屏幕的宽和高，实际上是不需要考虑分辨率的，因为在鼠标的坐标系下，电脑的宽和高都被分成了65535个单位，所以宽和高可以视为65535。根据这些，就可以算出![](http://images.cnblogs.com/cnblogs_com/xz816111/786501/o_x2y2.gif)的值来。


<br/><br/><br/><br/><font/>
