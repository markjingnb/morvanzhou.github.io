---
youku_id: XMTU4ODcwNjI1Ng
youtube_id: ttSUtDTjDyI
title: Numpy array 合并
description: 


date: 2016-11-3
chapter: 2
---
* 学习资料:
  * [相关代码]()

numpy array 都可以进行两个或多个 array 的横向或纵向的合并.

我们可以用到:
np.vstack((A,B))  # 纵向合并
np.stack((A,B))   # 横向合并
或者:
np.concatenate((A,B,B,A), axis=1) # 选择你要合并的 axis