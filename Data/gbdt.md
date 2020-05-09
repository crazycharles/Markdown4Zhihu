# 梯度提升树（GBDT）简介

网络上有关梯度提升树（Gradient Boosting Decision Tree, **GBDT**）的帖子其实已经比较繁杂了，但是考虑到Facebook的GBDT+LR方法在CTR预估中提供了一个重要的思路，为此还是有必要再学习一遍GBDT方法，加深自己理解的。这篇文章会梳理一下GBDT的原理，包括应对回归问题、二分类问题和多分类问题。但由于自己对GBDT的理解尚不十分透彻，所以文章中难免会出现一些错误，也烦请大家尽管一一指出，不胜感谢！

我发现很多讲提升树的文章、博客都是从针对回归问题的提升树模型开始讲的，但我觉得以平方误差（Square loss）为损失的回归问题仅仅是提升树模型可解决问题的一个特例，我反倒觉得应该先介绍GBDT，因为它才是提升树模型的通用形式。

## 1 前向分步算法与梯度提升

GBDT以及其他类型的提升树模型都是基于前向分步算法的（Forward stagewise algorithm）。而前向分步算法可以这样来理解，假设我们要使用决策树来预测一个人的年龄，刚开始的时候模型初始化会直接给出一个预测值$f_{0}(x)$，注意这个预测值不需要训练决策树来得到，而且不一定精确（比如刚开始模型初始预测为0岁，或者根据人群年龄分布给出一个相对合理的值）。接着在模型上一步所给出的预测基础上来训练第一棵决策树，此时模型的输出便是模型初始化的预测值加上第一棵决策树的输出值，然后我们继续添加第二棵决策树，使得第二棵决策树能够在前面所构造的模型基础之上，让损失最小，不断的进行这一过程直到构建的决策树棵数满足要求或者损失小于一个阈值。当进行到第$m$步时，模型可以表示为：
$$
f_{m}(x)=f_{m-1}(x)+\beta_{m}T(x;\Theta_{m}),\tag{1}
$$
其中$f_{m-1}(x)$是已经构造的$m-1$棵决策树所组成的模型（包括模型初始值），而$T(x;\Theta_{m})$是我们当前需要构造的第$m$棵决策树，$\beta_{m}$代表决策树的系数。需要提及一点的是，前向分步算法并不是拘泥于某一个分类器，此处仅以决策树为例。因为前面的$m-1$棵决策树已经训练好了，参数都已经固定了，当系数$\beta_{m}$取一个固定值时，那么在第$m$步，我们仅需要训练第$m$棵树的参数$\Theta_{m}$来最小化当前的损失：
$$
\hat{\Theta}_{m}=\mathop{\arg\min}_{\Theta_{m}}\sum_{i=1}^{N}L(y_{i},f_{m-1}(x_{i})+\beta_{m}T(x_{i};\Theta_{m})),\tag{2}
$$
其中$N$代表样本的总个数，$L$代表损失函数。在前向分步算法搞清楚之后，我们再来回顾一下机器学习中是如何最小化损失函数的。

假设现有一损失函数$J(\theta)$，当我们需要对这个参数为$\theta$的损失函数求最小值时，只要按照损失函数的负梯度方向调整参数$\theta$即可，因为梯度方向是使得函数增长最快的方向，沿着梯度的反方向也就是负梯度方向会使得函数减小最快，当学习率为$\rho$时，$\theta$的更新方法如下：
$$
\theta_{i}:=\theta_{i}-\rho\frac{\partial J}{\partial \theta_{i}}.\tag{3}
$$
那么同理，前向分步算法模型的损失$L$只和构造的模型$f_{m-1}(x)$有关，为了使得损失总体损失$J=\sum_{i}L(y_{i},f_{m-1}(x_{i}))$进一步降低，需要对$f_{m-1}(x)$求导，而$f_{m-1}(x)$是针对$N$个样本的，所以对每个样本预测值求偏导数：
$$
f_{m}(x_{i}):=f_{m-1}(x_{i})-\rho\frac{\partial J}{\partial f_{m-1}(x_{i})},\tag{4}
$$
这里的$\rho$和刚才所提到的$\beta_{m}$的作用是相同的，我们在此令$\rho =1$。因此，对于第$m$棵决策树而言，其拟合的不再是原有数据$(x_{i},y_{i})$的目标值$y_{i}$，而是负梯度，这样才能使得损失函数下降最快：
$$
\{(x_{1},-\frac{\partial J}{\partial f_{m-1}(x_{1})}),(x_{2},-\frac{\partial J}{\partial f_{m-1}(x_{2})}),\cdots,(x_{n},-\frac{\partial J}{\partial f_{m-1}(x_{n})})\},m=1,2,\cdots,M.\tag{5}
$$
到此为止，我们把GBDT中梯度提升的含义理清了，也就是说，模型开始时会给出一个初始预测值，在构建后续的决策树时，每个样本的目标值就变成了损失函数对当前模型在这个样本产生的负梯度，这样做的目的是为了最小化损失函数，进而提升模型性能。最后我们将所有的决策树组合起来，便是GBDT模型。

## 2 分类与回归树

在上一小节中，我们搞清楚了Gradient Boosting的含义，就是通过拟合梯度来提升模型性能，那么Decision Tree在GBDT中的具体形式是什么呢？本小节来介绍GBDT所使用的分类与回归树（Classification and regression tree, CART）。

从名字上看，CART决策树可以处理分类和回归问题，这当然取决于分裂准则的选择。当采用平方误差损失时，可以应对回归问题，当采用基尼指数指导分裂时，可以应对分类问题。但在GBDT采用CART决策树作为基函数时，尽管GBDT可以处理分类与回归问题（实际上这也取决于GBDT损失函数的选择），CART决策树的分裂准则一直采用平方误差损失（比如在sklearn的实现中，均默认采用“friedman_mse“，一种基于平方误差损失的改进版），因此在这一节中，我们主要介绍一下当损失函数为平方误差时CART决策树的生成流程。

---

<u>**算法1:CART生成流程**</u>

对给定训练集$D$

（1）依次遍历每个变量的取值，寻找最优切分变量$j$与切分点$s$，进而求解：
$$
\min_{j,s}\left[\min_{c_{1}}\sum_{x_{i}\in R_{1}(j,s)}(y_{i}-c_{1})^{2}+\min_{c_{2}}\sum_{x_{i}\in R_{2}(j,s)}(y_{i}-c_{2})^{2} \right].\tag{6}
$$
这一步就是对当前节点寻找最优切分点，使得切分之后两个节点总体平方误差最小。

（2）接着用选定的对$(j,s)$划分区域并决定相应的输出值：
$$
R_{1}(j,s)=\{x|x^{(j)}\leq s\},R_{2}(j,s)=\{x|x^{(j)}>s\},\tag{7}
$$

$$
\hat{c}_{m}=\frac{1}{N_{m}}\sum_{x_{i}\in R_{m}(j,s)}y_{i},x\in R_{m},m=1,2.\tag{8}
$$

$R_{1}(j,s),R_{2}(j,s)$分别代表的是左子节点和右子节点，而对两个节点上的估计值$\hat{c}_{m}$采用相应子节点上目标值的均值来表示（**注意这里划分区间时是一直按照平方误差损失最小划分的，但$\hat{c}_{m}$采用均值是因为此时的损失函数为平方误差，当损失函数变化时便不是均值了**）。

（3）继续对两个子区域调用步骤（1），（2），直到满足停止条件（比如子节点上样本个数过少或者决策树已经达到指定的深度）。

（4）将输入空间划分为$M$个区域$R_{1},R_{2},\cdots,R_{M}$，生成决策树：
$$
f(x)=\sum_{m=1}^{M}\hat{c}_{m}I(x\in R_{m}).\tag{9}
$$
$f(x)$便是我们学习到的CART决策树，$I(x\in R_{m})$表示相应样本属于的区域，在相应区域其值为1，否则为0。

---



## 3 处理回归问题的GBDT

在介绍了GBDT中Gradient Boosting和Decision Tree的含义之后，本小节会给出处理回归问题的GBDT形式。对于回归问题来说，我们以平方误差损失函数为例，来介绍GBDT的处理方法。
$$
L(y,f(x))=\frac{1}{2}\cdot(y-f(x))^{2},\tag{10}
$$
当损失函数如公式$(10)$所示时，对于总体的损失函数$J=\sum_{i}L(y_{i},f_{m-1}(x_{i}))$，其对每个样本当前预测值产生的梯度为：
$$
\frac{\partial J}{\partial f_{m-1}(x_{i})}=\frac{\partial \sum_{i}L(y_{i},f_{m-1}(x_{i}))}{\partial f_{m-1}(x_{i})}=\frac{\partial L(y_{i},f_{m-1}(x_{i}))}{\partial f_{m-1}(x_{i})}=f_{m-1}(x_{i})-y_{i},\tag{11}
$$
那么对应的负梯度：
$$
-\frac{\partial J}{\partial f_{m-1}(x_{i})}=y_{i}-f_{m-1}(x_{i}),\tag{12}
$$
其中$y_{i}-f_{m-1}(x_{i})$便是当前模型对数据需要拟合的残差（Residual），因此对于回归问题的提升树模型而言，每次所添加的基函数只需要去拟合上次遗留的残差即可。

所以回归问题的GBDT方法流程可以写为：

---

<u>**算法2:回归问题的GBDT流程**</u>

输入：训练数据集$T=\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{N},y_{N})\},x_{i}\in \mathcal{X}\subseteq\bf{R}^{n},y_{i}\in \mathcal{Y}\subseteq\bf{R}$，以及基函数系数或者学习率$\beta_{m}$。

输出：提升树$f_{M}(x)$。

（1）初始化$f_{0}(x)=\frac{1}{N}\sum_{i}^{N}y_{i}$（为什么这样初始化可以参考第6小节第1条）。

（2）对$m=1,2,\cdots,M$。

​	（a）计算残差：
$$
r_{mi}=y_{i}-f_{m-1}(x_{i}),i=1,2,\cdots,N.\tag{13}
$$
​	（b）拟合残差$r_{mi}$学习一个回归树，得到第$m$棵树的叶结点区域$R_{mj},j=1,2,\cdots,J$。**注意这一步是按照平方误差损失进行分裂节点的**。

​	（c）对$j=1,2,\cdots,J$，计算：
$$
c_{mj}=\arg\min\limits_{c}\sum\limits_{x_{i}\in R_{mj}}L(y_{i},f_{m-1}(x_{i})+c),\tag{14}
$$
​		需要注意的是，这一步是为了计算怎么对叶结点区域内的样本进行赋值，使得模型损失最小，因此这一步是需要根据当前模型的损失函数来计算的，当前模型是针对回归问题的模型，我们的损失函数是平方误差损失，所以这一步计算结果为（计算过程放在了第6小节第2条）：
$$
c_{mj}=\frac{1}{N_{mj}}\sum\limits_{x_{i}\in R_{mj}}r_{mi},\tag{15}
$$
即每个叶结点区域残差的均值。

​	（d）更新$f_{m}(x)=f_{m-1}(x)+\beta_{m}\sum\limits_{j=1}^{J}c_{mj}I(x\in R_{mj})$。

（3）得到回归问题提升树
$$
f_{M}(x)=\sum_{m=1}^{M}\beta_{m}\sum_{j=1}^{J}c_{mj}I(x\in R_{mj}).\tag{16}
$$

---

需要再提一遍的是，$\beta_{m}$可以设置为1，但它的作用和梯度下降时的学习率一致，能够使得损失函数收敛到更小的值，因此可以调小$\beta_{m}$，而一个较小的学习率会使得模型趋向于训练更多的决策树，使得模型更加稳定，不至于过拟合。

## 4 处理二分类问题的GBDT

因为处理回归问题的GBDT采用的是以平方误差为分裂准则的CART决策树，其叶节点的输出其实是整个实数范围的，我们可以参考逻辑回归在线性回归的基础上添加激活函数的方法，来使得GBDT同样能够应对分类问题。

参考之前的文章——[逻辑回归简介及实现](https://zhuanlan.zhihu.com/p/130209974)，我们可以把二分类GBDT的损失函数定义为：
$$
L(y,f(x))=-y\log \hat{y}-(1-y)\log(1-\hat{y}),\tag{17}
$$

$$
\hat{y}=\frac{1}{1+e^{-f(x)}},\tag{18}
$$

其中$\hat{y}$代表的是当前样本类别为1的概率$P(y=1|x)$。那么对于总体损失函数$J=\sum_{i}L(y_{i},f_{m-1}(x_{i}))$，其对每个样本当前预测值产生的梯度为：
$$
\begin{align}
\frac{\partial J}{\partial f_{m-1}(x_{i})} 
&= \frac{\partial L(y_{i},f_{m-1}(x_{i}))}{\partial f_{m-1}(x_{i})}\\
&= \frac{\partial \left[-y_{i}\log\frac{1}{1+e^{-f_{m-1}(x_{i})}}-(1-y_{i})\log(1-\frac{1}{1+e^{-f_{m-1}(x_{i})}})\right]}{\partial f_{m-1}(x_{i})}\\
&= \frac{\partial \left[y_{i}\log(1+e^{-f_{m-1}(x_{i})})+(1-y_{i})[f_{m-1}(x_{i})+\log(1+e^{-f_{m-1}(x_{i})})] \right]}{\partial f_{m-1}(x_{i})} \\
&= \frac{\partial \left[(1-y_{i})f_{m-1}(x_{i})+\log(1+e^{-f_{m-1}(x_{i})})\right]}{\partial f_{m-1}(x_{i})}\\
&= \frac{1}{1+e^{-f_{m-1}(x_{i})}}-y_{i}\\
&= \hat{y_{i}}-y_{i}
\end{align},\tag{19}
$$
也就是当前模型预测的概率值与真实样本之间的差值，那么负梯度可以写为：
$$
-\frac{\partial J}{\partial f_{m-1}(x_{i})}=y_{i}-\hat{y}_{i},\tag{20}
$$
因此对于处理二分类问题的GBDT来说，其每次拟合的是真实样本值与模型预测值的差值，所以二分类问题的GBDT方法流程可以写为：

---

<u>**算法3:二分类问题的GBDT流程**</u>

输入：训练数据集$T=\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{N},y_{N})\},x_{i}\in \mathcal{X}\subseteq\bf{R}^{n},y_{i}\in \{0,1\}$，以及基函数系数或者学习率$\beta_{m}$。

输出：提升树$f_{M}(x)$。

（1）初始化$f_{0}(x)=\log\frac{p_{1}}{1-p_{1}}$，其中$p_{1}$代表的是样本中$y=1$的比例（为什么这样初始化可以参考第6小节第3条）。

（2）对$m=1,2,\cdots,M$。

​	（a）计算负梯度：
$$
r_{mi}=y_{i}-\hat{y}_{i},i=1,2,\cdots,N.\tag{21}
$$
​	其中$\hat{y}_{i}=\frac{1}{1+e^{-f_{m-1}(x_{i})}}$代表$y_{i}=1$的概率。

​	（b）拟合负梯度$r_{mi}$学习一个回归树，得到第$m$棵树的叶结点区域$R_{mj},j=1,2,\cdots,J$。**注意这一步也是按照平方误差损失进行分裂节点的**。

​	（c）对$j=1,2,\cdots,J$，计算：
$$
c_{mj}=\arg\min\limits_{c}\sum\limits_{x_{i}\in R_{mj}}L(y_{i},f_{m-1}(x_{i})+c),\tag{22}
$$
​		同样的，当前模型是针对二分类问题的模型，我们的损失函数是交叉熵损失，所以这一步计算结果为（计算过程请见第6小节第4条）：
$$
c_{mj}=\frac{\sum\limits_{x_{i}\in R_{mj}}r_{mi}}{\sum\limits_{x_{i}\in R_{mj}}(y_{i}-r_{mi})(1-y_{i}+r_{mi})}.\tag{23}
$$
​	（d）更新$f_{m}(x)=f_{m-1}(x)+\beta_{m}\sum\limits_{j=1}^{J}c_{mj}I(x\in R_{mj})$。

（3）得到二分类问题提升树
$$
f_{M}(x)=\sum_{m=1}^{M}\beta_{m}\sum_{j=1}^{J}c_{mj}I(x\in R_{mj}).\tag{24}
$$

---



## 5 处理多分类问题的GBDT

二分类问题对于GBDT来讲，采取了和逻辑回归同样的方式，即利用sigmoid函数将输出值归一化。而在处理多分类问题时，GBDT可以采用softmax函数一步步求解。

需要说明的是，对于多分类问题每个样本的类标都要先经过one-hot编码，这样才能划分成多个二分类问题来求解。假设数据集$D$的类别数$K=4$，样本个数为$N$，并且样本$x_{1},x_{2},x_{3},x_{4}$的类标分别为$y_{1}=0,y_{2}=1,y_{3}=2,y_{4}=3$，那么在处理多分类问题时，GBDT将会通过one-hot编码将原有的一个多分类问题转化为四个二分类问题，如下表所示：

| $x_{i}$   | $x_{1}$     | $x_{2}$     | $x_{3}$     | $x_{4}$     | $F_{0}(x)$ | $F_{1}(x)$          | $F_{2}(x)$          | $\cdots$ | $F_{M}(x)$          |
| :-------- | :---------- | :---------- | :---------- | :---------- | :--------- | :------------------ | :------------------ | :------- | :------------------ |
| $y_{i,1}$ | $y_{1,1}=1$ | $y_{2,1}=0$ | $y_{3,1}=0$ | $y_{4,1}=0$ | 0          | $T_{1,1}(x,\Theta)$ | $T_{1,2}(x,\Theta)$ | $\cdots$ | $T_{1,M}(x,\Theta)$ |
| $y_{i,2}$ | $y_{1,2}=0$ | $y_{2,2}=1$ | $y_{3,2}=0$ | $y_{4,2}=0$ | 0          | $T_{2,1}(x,\Theta)$ | $T_{2,2}(x,\Theta)$ | $\cdots$ | $T_{2,M}(x,\Theta)$ |
| $y_{i,3}$ | $y_{1,3}=0$ | $y_{2,3}=0$ | $y_{3,3}=1$ | $y_{4,3}=0$ | 0          | $T_{3,1}(x,\Theta)$ | $T_{3,2}(x,\Theta)$ | $\cdots$ | $T_{3,M}(x,\Theta)$ |
| $y_{i,4}$ | $y_{1,4}=0$ | $y_{2,4}=0$ | $y_{3,4}=0$ | $y_{4,4}=1$ | 0          | $T_{4,1}(x,\Theta)$ | $T_{4,2}(x,\Theta)$ | $\cdots$ | $T_{4,M}(x,\Theta)$ |

由于我在表里添加了很多内容，所以这里再进行一些说明，

1. one-hot编码的意思是每个样本的类别都会被转为一个$K$维向量，并且向量中仅有一个值为1，比如$y_{2}=1$被转为$\{0,1,0,0\}$；
2. 在转换之后，$y_{i,j}$代表的是第$i$个样本在第$j$个二分类问题的表示值，比如$y_{3,2}=0$代表的是第3个样本$x_{3}$在第2个二分类问题中类标变为0，其中$i=1,2,\cdots,N$，$j=1,2,\cdots,K$。
3. $F_{0}(x)=0$是模型预测的初始值，后续会提到，还可以采用其他的初始化方法。
4. $F_{1}(x)$代表的是进行第1轮学习后的模型，由$F_{0}(x)+\{T_{1,1}(x,\Theta),T_{2,1}(x,\Theta),T_{3,1}(x,\Theta),T_{4,1}(x,\Theta)\}$共同组成的，同理$F_{M}(x)$代表的是经过$M$轮学习后的最终GBDT模型，由$F_{M-1}(x)+\{T_{1,M}(x,\Theta),T_{2,M}(x,\Theta),T_{3,M}(x,\Theta),T_{4,M}(x,\Theta)\}$。
5. $T_{j,m}(x,\Theta)$代表的是对第$j$个二分类问题进行第$m$轮学习时，拟合负梯度所产生的决策树，其中$j=1,2,\cdots,K$，$m=1,2,\cdots,M$。比如$T_{4,2}(x,\Theta)$代表的是针对第4个二分类问题，进行第2次学习之后产生的决策树，并且其训练样本$x_{1},x_{2},x_{3},x_{4}$的类标为$\{0,0,0,1\}$，当然这是原始的类别，$T_{4,2}(x,\Theta)$在学习的时候，样本类标已经变为基于上轮学习产生的负梯度了。

当进行了$m$轮学习之后，GBDT模型变为$F_{m}(x)$，那么对于样本$x$，其属于每个类别$j$的概率可以预测为：
$$
P(y=j|x)=\frac{e^{F_{j,m}(x)}}{\sum_{l=1}^{K}e^{F_{l,m}(x)}},j=1,2,\cdots,K,\tag{25}
$$
其中$F_{j,m}(x)$代表的是模型经过$m$轮学习之后，在第$j$类二分类问题中，对当前样本$x$的预测值。

对于多分类问题的GBDT来说，在经过$m$轮后，模型$F_{m}(x)$对于第$i$个样本的损失函数为：
$$
L(y_{i,*},F_{m}(x_{i}))=-\sum_{j=1}^{K}y_{i,j}\log P(y_{i,j}|x_{i})=-\sum_{j=1}^{K}y_{i,j}\log\frac{e^{F_{j,m}(x_{i})}}{\sum_{l=1}^{K}e^{F_{l,m}(x_{i})}},\tag{26}
$$
推导出损失函数的详细过程请参考第6小节第5条。

还是老步骤，接下来我们求出单个样本损失函数的负梯度$-\frac{\partial L(y_{i,*},F_{m}(x_{i}))}{\partial F_{k,m}(x_{i})}$，但在那之前，为了方便后续展示，在这里我们先求出$\frac{\partial P(y_{i,j}|x_{i})}{\partial F_{k,m}(x_{i})}$：
$$
\begin{align}
\frac{\partial P(y_{i,j}|x_{i})}{\partial F_{k,m}(x_{i})}
&=\frac{\partial \frac{e^{F_{j,m}(x_{i})}}{\sum_{l=1}^{K}e^{F_{l,m(x_{i})}}}}{\partial F_{k,m}(x_{i})}\\
&=\frac{\frac{\partial e^{F_{j,m}(x_{i})}}{\partial F_{k,m}(x_{i})}\sum_{l=1}^{K}e^{F_{l,m}(x_{i})}-e^{F_{j,m}(x_{i})}\frac{\partial \sum_{l=1}^{K}e^{F_{l,m}(x_{i})}}{\partial F_{k,m}(x_{i})}}
{[\sum_{l=1}^{K}e^{F_{l,m(x_{i})}}]^{2}},
\end{align}\tag{27}
$$
这里分为两种情况求解，第一种当$j=k$时：
$$
\frac{\partial e^{F_{j,m}(x_{i})}}{\partial F_{k,m}(x_{i})}=e^{F_{j,m}(x_{i})},\tag{28}
$$
因而：
$$
\begin{align}
\frac{\partial P(y_{i,j}|x_{i})}{\partial F_{k,m}(x_{i})}
&=\frac{e^{F_{j,m}(x_{i})}\sum_{l=1}^{K}e^{F_{l,m}(x_{i})}-e^{F_{j,m}(x_{i})}e^{F_{k,m}(x_{i})}}{[\sum_{l=1}^{K}e^{F_{l,m(x_{i})}}]^{2}}\\
&=P(y_{i,j}|x_{i})\cdot[1-P(y_{i,k}|x_{i})].
\end{align}\tag{29}
$$


第二种情况，即当$j\neq k$时：
$$
\frac{\partial e^{F_{j,m}(x_{i})}}{\partial F_{k,m}(x_{i})}=0.\tag{30}
$$
因而：
$$
\begin{align}
\frac{\partial P(y_{i,j}|x_{i})}{\partial F_{k,m}(x_{i})}
&=\frac{-e^{F_{j,m}(x_{i})}e^{F_{k,m}(x_{i})}}{[\sum_{l=1}^{K}e^{F_{l,m(x_{i})}}]^{2}}\\
&=-P(y_{i,j}|x_{i})\cdot P(y_{i,k}|x_{i}).
\end{align}\tag{31}
$$
回到我们要求解的梯度：
$$
\begin{align}
\frac{\partial L(y_{i,*},F_{m}(x_{i}))}{\partial F_{k,m}(x_{i})}
&=\frac{\partial-\sum_{j=1}^{K}y_{i,j}\log P(y_{i,j}|x_{i})}{\partial F_{k,m}(x_{i})}\\
&=-\sum_{j=1}^{K}y_{i,j}\frac{1}{P(y_{i,j}|x_{i})}\frac{\partial P(y_{i,j}|x_{i})}{\partial F_{k,m}(x_{i})}\\
&=-y_{i,k}\frac{1}{P(y_{i,k}|x_{i})}P(y_{i,k}|x_{i})\cdot[1-P(y_{i,k}|x_{i})]-\sum_{j\neq k}^{K}y_{i,j}\frac{1}{P(y_{i,j}|x_{i})}[-P(y_{i,j}|x_{i})\cdot P(y_{i,k}|x_{i})]\\
&=-y_{i,k}+y_{i,k}P(y_{i,k}|x_{i})+\sum_{j\neq k}^{K}y_{i,j}P(y_{i,k}|x_{i})\\
&=-y_{i,k}+\sum_{j=1}^{K}y_{i,j}P(y_{i,k}|x_{i})\\
&=-y_{i,k}+P(y_{i,k}|x_{i})\sum_{j=1}^{K}y_{i,j}\\
&=P(y_{i,k}|x_{i})-y_{i,k},
\end{align}\tag{32}
$$
因此所求的单样本损失函数的负梯度：
$$
-\frac{\partial L(y_{i,*},F_{m}(x_{i}))}{\partial F_{k,m}(x_{i})}=y_{i,k}-P(y_{i,k}|x_{i}),\tag{33}
$$
对于处理多分类问题的GBDT来说，其每次拟合的是真实类标与模型预测值之差，所以多分类问题的GBDT方法流程可以写为：

---

<u>**算法4:多分类问题的GBDT流程**</u>

输入：训练数据集$T=\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{N},y_{N})\},x_{i}\in \mathcal{X}\subseteq\bf{R}^{n},y_{i}\in \{0,1,\cdots,K\}$，以及基函数系数或者学习率$\beta_{m}$。

输出：提升树$f_{M}(x)$。

（1）初始化$F_{k,0}(x)=0$，即$P(y_{i,k}|x)=\frac{1}{K},k=1,2,\cdots,K$。这样初始化是考虑到当我们对数据一无所知时，最好的假设就是每类发生的概率相等，但当我们知道数据集类别分布情况后，我们可以按照类别比例来设置初始化。

（2）对$m=1,2,\cdots,M$。

​	（a）对$i=1,2,\cdots,N,k=1,2,\cdots,K$，计算 $P(y_{i,k}|x_{i})=\frac{e^{F_{k,m-1}(x_{i})}}{\sum_{l=1}^{K}e^{F_{l,m-1}(x_{i})}}$。

​	（b）对$k=1,2,\cdots,K$：

​		（b1）计算负梯度：
$$
r_{i,k,m}=y_{i,k}-P(y_{i,k}|x_{i}),i=1,2,\cdots,N.\tag{34}
$$
​			其中$r_{i,k,m}$代表的是模型在第$m$轮学习中，第$k$个二分类问题里，对第$i$个样本产生的负梯度。

​		（b2）拟合负梯度$r_{i,k,m}$学习一个回归树，得到第$m$棵树的叶结点区域$R_{j,k,m},j=1,2,\cdots,J$。**注意这一步也是按照平方误差损失进行分裂节点的**。

​		（b3）对$j=1,2,\cdots,J$，计算：
$$
c_{j,k,m}=\arg\min\limits_{c}\sum\limits_{x_{i}\in R_{j,k,m}}L(y_{i,k},F_{k,m-1}(x_{i})+c),\tag{35}
$$


​			同样的，当前模型是针对多分类问题的模型，根据多分类模型的损失函数，这一步计算结果为（这一步我没有推导出来，我自己按照泰勒二阶展开推导的结果却是$c_{j,k,m}=\frac{1}{P(y_{i,k}|x_{i})}$，感兴趣的同学可以看一下论文原文，也就是参考文献5，也希望了解这一步的大佬在评论区指出正确解法）：
$$
c_{j,k,m}=\frac{K-1}{K}\frac{\sum_{x_{i}\in R_{j,k,m}}r_{i,k,m}}{\sum_{x_{i}\in R_{j,k,m}}|r_{i,k,m}|(1-|r_{i,k,m}|)}.\tag{36}
$$
​		（b4）更新$F_{k,m}(x)=F_{k,m-1}(x)+\beta_{m}\sum\limits_{j=1}^{J}c_{j,k,m}I(x\in R_{j,k,m})$。

（3）得到多分类问题提升树
$$
F_{k,M}(x)=\sum_{m=1}^{M}\beta_{m}\sum_{j=1}^{J}c_{j,k,m}I(x\in R_{j,k,m}).\tag{37}
$$

---



## 6 公式推导

### （1）对于回归问题的GBDT模型，为什么初始化为$f_{0}(x)=\frac{1}{N}\sum_{i}^{N}y_{i}$？

由于回归问题的损失函数为平方误差损失$J=\sum_{i}^{N}L(y_{i},f(x_{i})=\sum_{i}^{N}\frac{1}{2}\cdot(y_{i}-f(x_{i}))^2$，为了使得损失函数最小化，那么我们在初始化的时候也希望能够有一个好的起点，即希望能够进行一个使得损失函数最小的初始化操作：
$$
\begin{align}
\frac{\partial J}{\partial f_{0}(x)}
&=\frac{\partial \sum_{i}^{N}\frac{1}{2}\cdot(y_{i}-f_{0}(x))^{2}}{\partial f_{0}(x)}\\
&=\frac{\partial \sum_{i}^{N}\left[\frac{1}{2}f_{0}^{2}(x)-y_{i}f_{0}(x)\right]}{\partial f_{0}(x)}\\
&=\sum_{i}^{N}(f_{0}(x)-y_{i})
\end{align}\tag{38}
$$
令$\frac{\partial J}{\partial f_{0}(x)}=0$可得，$f_{0}(x)=\frac{1}{N}\sum_{i}^{N}y_{i}$。

### （2）对于回归问题的GBDT，叶结点输出值$c_{mj}$的计算过程。

由于$c_{mj}=\arg\min\limits_{c}\sum\limits_{x_{i}\in R_{mj}}L(y_{i},f_{m-1}(x_{i})+c)$，那么为了使损失函数最小，我们令$\frac{\partial L}{\partial c}=0$即可，
$$
\begin{align}
\frac{\partial L}{\partial c}
&=\frac{\partial\sum\limits_{x_{i}\in R_{mj}}L(y_{i},f_{m-1}(x_{i})+c)}{\partial c}\\
&=\frac{\partial\sum\limits_{x_{i}\in R_{mj}}\frac{1}{2}(y_{i}-f_{m-1}(x_{i})-c)^{2}}{\partial c}\\
&=\sum\limits_{x_{i}\in R_{mj}}\left[c-(y_{i}-f_{m-1}(x_{i}))\right]
\end{align}\tag{39}
$$
由$\frac{\partial L}{\partial c}=0$得：
$$
\begin{align}
c_{mj}
&=\frac{1}{N_{mj}}\sum\limits_{x_{i}\in R_{mj}}(y_{i}-f_{m-1}(x_{i}))\\
&=\frac{1}{N_{mj}}\sum\limits_{x_{i}\in R_{mj}}r_{mi},
\end{align}\tag{40}
$$
其中$N_{mj}$代表的是第$m$棵决策树的第$j$个叶结点区域包含的样本数量。

### （3）对于二分类问题的GBDT，为什么初始化为$f_{0}(x)=\log\frac{p_{1}}{1-p_{1}}$？其中$p_{1}$代表的是样本中$y=1$的比例。

和回归问题GBDT初始化操作一样，这样初始化是为了最小化损失函数。对于二分类问题的GBDT，其损失函数为：
$$
\begin{align}
J
&=\sum_{i=1}^{N}L(y_{i},f(x_{i}))\\
&=\sum_{i=1}^{N}[-y_{i}\log \hat{y}_{i}-(1-y_{i})\log(1-\hat{y}_{i})]\\
&=\sum_{i=1}^{N}[-y_{i}\log \frac{1}{1+e^{-f(x_{i})}}-(1-y_{i})\log(1-\frac{1}{1+e^{-f(x_{i})}})].
\end{align}\tag{41}
$$
那么对于初始化函数$f_{0}(x)$，我们将损失函数对其求导：
$$
\begin{align}
\frac{\partial J}{\partial f_{0}(x)}
&=\frac{\partial\sum_{i=1}^{N}[-y_{i}\log \frac{1}{1+e^{-f_{0}(x_{i})}}-(1-y_{i})\log(1-\frac{1}{1+e^{-f_{0}(x_{i})}})]}{\partial f_{0}(x)}\\
&=\frac{\partial \sum_{i=1}^{N}[(1-y_{i})f_{0}(x_{i})+\log(1+e^{-f_{0}(x_{i})})]}{\partial f_{0}(x)}\\
&=\sum_{i=1}^{N}(1-y_{i}-\frac{e^{-f_{0}(x_{i})}}{1+e^{-f_{0}(x_{i})}}).
\end{align}\tag{42}
$$
注意其中的$f_{0}(x_{i})$对所有的$i=1,2,\cdots,N$都是一个常数值，最后我们令$\frac{\partial{J}}{\partial f_{0}(x)}=0$，在训练集中不全为负样本或全为正样本的情况下，$f_{0}(x)=\log\frac{\sum_{i=1}^{N}y_{i}}{N-\sum_{i=1}^{N}y_{i}}$，也就是$f_{0}(x)=\log\frac{p_{1}}{1-p_{1}}$，$p_{1}$为训练集中正样本的比例。

### （4）对于二分类问题的GBDT，叶结点$c_{mj}$输出值的计算过程。

在进行第$m$轮学习时，损失函数为$J=\sum\limits_{x_{i}\in R_{mj}}L(y_{i},f_{m-1}(x_{i})+c)$，由泰勒公式得：
$$
\begin{align}
J
&=\sum\limits_{x_{i}\in R_{mj}}L(y_{i},f_{m-1}(x_{i})+c)\\
&\approx \sum\limits_{x_{i}\in R_{mj}}[L(y_{i},f_{m-1}(x_{i}))+\frac{\partial L(y_{i},f_{m-1}(x_{i}))}{\partial f_{m-1}(x_{i})}\cdot c+\frac{1}{2}\frac{\partial^{2}L(y_{i},f_{m-1}(x_{i}))}{\partial^{2}f_{m-1}(x_{i})}\cdot c^{2}],
\end{align}\tag{43}
$$
而
$$
\frac{\partial L(y_{i},f_{m-1}(x_{i}))}{\partial f_{m-1}(x_{i})}= \frac{1}{1+e^{-f_{m-1}(x_{i})}}-y_{i},\tag{44}
$$

$$
\frac{\partial^{2}L(y_{i},f_{m-1}(x_{i}))}{\partial^{2}f_{m-1}(x_{i})}=\frac{e^{-f_{m-1}(x_{i})}}{[1+e^{-f_{m-1}(x_{i})}]^2}.\tag{45}
$$

那么损失函数可以写为（额，公式太长了）：
$$
J\approx \sum\limits_{x_{i}\in R_{mj}}\left[[-y_{i}\log \frac{1}{1+e^{-f_{m-1}(x_{i})}}-(1-y_{i})\log(1-\frac{1}{1+e^{-f(x_{i})}})]+[\frac{1}{1+e^{-f_{m-1}(x_{i})}}-y_{i}]\cdot c+\frac{1}{2}\frac{e^{-f_{m-1}(x_{i})}}{[1+e^{-f_{m-1}(x_{i})}]^2}\cdot c^{2}\right].\tag{46}
$$
所以求导可得：
$$
\begin{align}
\frac{\partial J}{\partial c}
&=\frac{\partial \sum\limits_{x_{i}\in R_{mj}}\left[[\frac{1}{1+e^{-f_{m-1}(x_{i})}}-y_{i}]\cdot c+\frac{1}{2}\frac{e^{-f_{m-1}(x_{i})}}{[1+e^{-f_{m-1}(x_{i})}]^2}\cdot c^{2}\right]}{\partial c}\\
&=\sum\limits_{x_{i}\in R_{mj}}\left[\frac{1}{1+e^{-f_{m-1}(x_{i})}}-y_{i}+\frac{e^{-f_{m-1}(x_{i})}}{[1+e^{-f_{m-1}(x_{i})}]^2}\cdot c\right].
\end{align}\tag{47}
$$
令$\frac{\partial J}{\partial c}=0$得：
$$
c_{mj}=\frac{\sum\limits_{x_{i}\in R_{mj}}r_{mi}}{\sum\limits_{x_{i}\in R_{mj}}(y_{i}-r_{mi})(1-y_{i}+r_{mi})}.\tag{48}
$$

### （5）多分类问题的GBDT，经过$m$轮学习，第$i$个样本的损失函数推导过程。

通过回顾之前的文章，可以知道逻辑回归的损失函数是通过极大似然估计法得来的，逻辑回归中的条件概率为：
$$
P(Y|x)=P(Y=1|x)^{y}P(Y=0|x)^{1-y},\tag{49}
$$
那么对于上文表格中的第$1$个样本：

| $x_{i}$   | $x_{1}$ |
| --------- | ------- |
| $y_{i,1}$ | 1       |
| $y_{i,2}$ | 0       |
| $y_{i,3}$ | 0       |
| $y_{i,4}$ | 0       |

来说，此时的条件概率可以写为：
$$
P(Y|x_{1})=P(y_{1,1}=1|x_{1})^{y_{1,1}}\cdot P(y_{1,2}=1|x_{1})^{y_{1,2}}\cdot P(y_{1,3}=1|x_{1})^{y_{1,3}}\cdot P(y_{1,4}=1|x_{1})^{y_{1,4}},\tag{50}
$$
所以处理多分类问题的GBDT中单个样本的条件概率可以写为：
$$
\begin{align}
P(Y|x_{i})
&=\prod_{j=1}^{K}P(y_{i,j}|x_{i})^{y_{i,j}}\\
&=\prod_{j=1}^{K}(\frac{e^{F_{j,m}(x)}}{\sum_{l=1}^{K}e^{F_{l,m}(x)}})^{y_{i,j}},j=1,2,\cdots,K.
\end{align}\tag{51}
$$
对于单个样本来说，我们希望其条件概率最大化，但损失函数往往要求最小化，所以对条件概率求负对数即可得到经过$m$轮训练之后，模型对单个样本的损失函数：
$$
L(y_{i,*},F_{m}(x_{i}))=-\sum_{j=1}^{K}y_{i,j}\log P(y_{i,j}|x_{i})=-\sum_{j=1}^{K}y_{i,j}\log\frac{e^{F_{j,m}(x_{i})}}{\sum_{l=1}^{K}e^{F_{l,m}(x_{i})}}.\tag{52}
$$

## 7. 总结

总的来说，这篇文章主要介绍了GBDT如何处理回归问题、二分类问题和多分类问题，但是还是有很多不足，比如多分类问题叶结点输出值的推导还不清楚，而且每类问题缺乏实例的解释。另外，我觉得掌握任何一种算法不光要做到能向其他人讲清楚原理，还要能够用代码实现它，但因为这个专栏的文章主要是学习CTR预估，后续还要学习很多的内容，因此代码实现在这里先留个坑，等我过一遍CTR预估经典方法之后再来完成。

感谢你能读到这里，毕竟我觉得这篇文章写的像坨$*$一样。

# 参考

1. [GBDT算法用于分类问题](https://zhuanlan.zhihu.com/p/46445201)

2. [GBDT原理与实践-多分类篇](https://blog.csdn.net/qq_22238533/article/details/79199605)

3. [Softmax函数与交叉熵](https://blog.csdn.net/behamcheung/article/details/71911133)

4. [GBDT--分类篇](https://blog.csdn.net/On_theway10/article/details/83576715)

5. [Greedy function approximation: a gradient boosting machine](https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451)

   



