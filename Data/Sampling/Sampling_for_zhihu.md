# Ray tracing 中的 sampling

（注：本文基本完全抄袭 [PBRT](http://www.pbr-book.org/3ed-2018/contents.html)）

## 方格子取点

将场景渲染为图像本质上是求解


<img src="https://www.zhihu.com/equation?tex=f(x, y, t, u, v, ...) \rightarrow L" alt="f(x, y, t, u, v, ...) \rightarrow L" class="ee_img tr_noresize" eeimg="1">

其中  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  为 2D 图像中的位置，也是发射出的光线的起点。如果  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  只取离散的整数点，那么会由于 sampling rate 过低 ~~导致sampling rate < 2 x 最大频率（Nyquist limit），进而~~ 导致 aliasing。要么通过 prefilter 除掉高频，要么提高 sampling rate。我们通常在每个整数像素对应的格子内多取几个点来发射光线（取点的数量叫做 samples per pixel（SPP））来提高 sampling rate。

但由于上文提到的  <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1">  在实际场景中最大频率  <img src="https://www.zhihu.com/equation?tex=\omega_0" alt="\omega_0" class="ee_img tr_noresize" eeimg="1">  为  <img src="https://www.zhihu.com/equation?tex=\infty" alt="\infty" class="ee_img tr_noresize" eeimg="1">  （几何体的边界会有一个值的突变），所以即使增大 sampling rate 也不能完全解决 aliasing 的问题。我们一般在取点时使用非均匀采样，使用一些随机因子，而不只是均匀间隔地取（例如  <img src="https://www.zhihu.com/equation?tex=(\frac{1}{4}, \frac{1}{4})" alt="(\frac{1}{4}, \frac{1}{4})" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=(\frac{1}{2}, \frac{1}{4})" alt="(\frac{1}{2}, \frac{1}{4})" class="ee_img tr_noresize" eeimg="1">  这样每隔  <img src="https://www.zhihu.com/equation?tex=\frac{1}{4}" alt="\frac{1}{4}" class="ee_img tr_noresize" eeimg="1">  取点），这样 sampling rate 不再固定，也就会减少固定的 aliasing，但是会产生噪音（noise），据说相比之下观感更好。

借助 [pcg-c](https://github.com/imneme/pcg-c)，我们可以生成  <img src="https://www.zhihu.com/equation?tex=[0, 1)" alt="[0, 1)" class="ee_img tr_noresize" eeimg="1">  范围内的随机数  <img src="https://www.zhihu.com/equation?tex=\xi" alt="\xi" class="ee_img tr_noresize" eeimg="1"> ：

```cpp
std::uint32_t GenUint32() {
    return pcg32_random_r(&pcg_rand_state_);
}
float GenFloat01() {
    return std::min(ldexp(static_cast<float>(GenUint32()), -32), kOneMinusEpsilon);
}
float Gen1D() {
    return GenFloat01();
}
glm::vec2 Gen2D() {
    return { GenFloat01(), GenFloat01() };
}
```

效果如下：

![](https://raw.githubusercontent.com/LYP951018/Markdown4Zhihu/master/Data/Sampling/square_plain.png)

为了使得生成的点覆盖得更广、更均匀，通常在完全随机的基础上使用 stratified Sampling，即预先将形状均匀分成若干个小格子（strata），在每个小格子内再随机采样，如下图所示。

![](https://raw.githubusercontent.com/LYP951018/Markdown4Zhihu/master/Data/Sampling/stratified_demo.png)

代码大致如下

```cpp
std::vector<glm::vec2> StratifiedSample2D(int nx, int ny, Sampler& sampler) {
    float dx = 1.0f / nx;
    float dy = 1.0f / ny;
    std::vector<glm::vec2> result(nx * ny);
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i) 
        {
            result[j * nx + i] = glm::min(
                (glm::vec2{ i, j } + sampler.Gen2D()) * glm::vec2{ dx, dy },
                glm::vec2{ 1.0f, 1.0f });
        }
    }
    return result;
}
```

效果如图

![](https://raw.githubusercontent.com/LYP951018/Markdown4Zhihu/master/Data/Sampling/square_stratified.png)

（似乎比刚才更均匀一些（x


## rejection sampling

在正方形中随机取样是挺简单，但是在 ray tracing 中，不仅仅需要在正方形内随机取点。例如使用 Thinlens 相机模型时，需要在 lens （即圆）上随机取点  <img src="https://www.zhihu.com/equation?tex=(u, v)" alt="(u, v)" class="ee_img tr_noresize" eeimg="1"> 。最朴素的方法是 rejection sampling：我们先调用在正方形里生成随机点的函数 `Sampler::Gen2D`，如果生成的点不在圆内，就再次调用 `Sampler::Gen2D`，直到生成的点在圆内。代码如下：

```cpp
glm::vec2 RejectionSampleDisk(Sampler& sampler) {
    glm::vec2 p;
    do {
        p = 2.0f * sampler.Gen2D() - glm::vec2{ 1.0f, 1.0f };
    } while (glm::length2(p) > 1.0f);
    return p;
}
```

![](https://raw.githubusercontent.com/LYP951018/Markdown4Zhihu/master/Data/Sampling/rejection_disk.png)


## Monte Carlo

渲染时往往需要使用积分，例如反射等式：


<img src="https://www.zhihu.com/equation?tex=L_o(p, \omega_o) = \int_{S^2} f(p, \omega_o, \omega_i)L_i(p, \omega_i) \cos\theta_i \,d\omega_i " alt="L_o(p, \omega_o) = \int_{S^2} f(p, \omega_o, \omega_i)L_i(p, \omega_i) \cos\theta_i \,d\omega_i " class="ee_img tr_noresize" eeimg="1">

其中， <img src="https://www.zhihu.com/equation?tex=L_o" alt="L_o" class="ee_img tr_noresize" eeimg="1">  为在表面上点  <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1">  出射方向为  <img src="https://www.zhihu.com/equation?tex=\omega_o" alt="\omega_o" class="ee_img tr_noresize" eeimg="1">  的 exitant radiance； <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1">  为表面的 BRDF； <img src="https://www.zhihu.com/equation?tex=L_i" alt="L_i" class="ee_img tr_noresize" eeimg="1">  为 incident radiance； <img src="https://www.zhihu.com/equation?tex=\theta_i" alt="\theta_i" class="ee_img tr_noresize" eeimg="1">  为  <img src="https://www.zhihu.com/equation?tex=\omega_o" alt="\omega_o" class="ee_img tr_noresize" eeimg="1">  与表面上点  <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1">  的法线的夹角。

但是计算机求不了这个积分，所以我们使用 Monte Carlo 来随机取几个变量来模拟取积分：


<img src="https://www.zhihu.com/equation?tex=F_N = \frac{1}{N}\int_{i=1}^{N} \frac{f(X_i)}{p(X_i)}" alt="F_N = \frac{1}{N}\int_{i=1}^{N} \frac{f(X_i)}{p(X_i)}" class="ee_img tr_noresize" eeimg="1">

其中，随机变量的  <img src="https://www.zhihu.com/equation?tex=X_i" alt="X_i" class="ee_img tr_noresize" eeimg="1">  的 PDF 为  <img src="https://www.zhihu.com/equation?tex=p(X_i)" alt="p(X_i)" class="ee_img tr_noresize" eeimg="1">  。 <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1">  的形状越和  <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1">  相似，那么  <img src="https://www.zhihu.com/equation?tex=F_N" alt="F_N" class="ee_img tr_noresize" eeimg="1">  就会越接近要求的积分值。所以我们从  <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1">  找到一个接近其形状的  <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1"> ，然后按照这个  <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1">  来取几个变量  <img src="https://www.zhihu.com/equation?tex=X_i" alt="X_i" class="ee_img tr_noresize" eeimg="1"> 。就像上面的等式，我们在选取随机的  <img src="https://www.zhihu.com/equation?tex=\omega_o" alt="\omega_o" class="ee_img tr_noresize" eeimg="1">  时，可能希望  <img src="https://www.zhihu.com/equation?tex=p(\omega)" alt="p(\omega)" class="ee_img tr_noresize" eeimg="1">  满足


<img src="https://www.zhihu.com/equation?tex=p(\omega) \propto \cos\theta" alt="p(\omega) \propto \cos\theta" class="ee_img tr_noresize" eeimg="1">

其中， <img src="https://www.zhihu.com/equation?tex=\omega" alt="\omega" class="ee_img tr_noresize" eeimg="1">  为选的方向， <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  为表面上的点与球圆心连成的线与 z 轴的夹角。

总而言之，我们希望能够生成 PDF 为指定的  <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1">  的随机变量。


## Inversion method

假设我们想分别按 PDF 为  <img src="https://www.zhihu.com/equation?tex=p_1" alt="p_1" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=p_2" alt="p_2" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=p_3" alt="p_3" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=p_4" alt="p_4" class="ee_img tr_noresize" eeimg="1">  来采样四个随机变量，我们把它们的 CDF （ <img src="https://www.zhihu.com/equation?tex=\sum_{1}^{i} p_i" alt="\sum_{1}^{i} p_i" class="ee_img tr_noresize" eeimg="1"> ） 绘制出来，如图所示：

![](https://raw.githubusercontent.com/LYP951018/Markdown4Zhihu/master/Data/Sampling/discrete-cdf.svg)

如果我们在  <img src="https://www.zhihu.com/equation?tex=[0, 1)" alt="[0, 1)" class="ee_img tr_noresize" eeimg="1">  随机取变量  <img src="https://www.zhihu.com/equation?tex=\xi" alt="\xi" class="ee_img tr_noresize" eeimg="1"> ，在 Y 轴画条线找到第一个有交点的变量，就显而易见地满足了指定的 PDF :

![](https://raw.githubusercontent.com/LYP951018/Markdown4Zhihu/master/Data/Sampling/discrete-inversion.svg)

将这个方法扩展到连续的随机变量上为：

- 求指定的 PDF  <img src="https://www.zhihu.com/equation?tex=p" alt="p" class="ee_img tr_noresize" eeimg="1">  的 CDF  <img src="https://www.zhihu.com/equation?tex=P(x) = \int_{0}^{x} p(x')\,dx'" alt="P(x) = \int_{0}^{x} p(x')\,dx'" class="ee_img tr_noresize" eeimg="1"> ；
- 求  <img src="https://www.zhihu.com/equation?tex=P^{-1}(x)" alt="P^{-1}(x)" class="ee_img tr_noresize" eeimg="1"> ；
- 在  <img src="https://www.zhihu.com/equation?tex=[0, 1)" alt="[0, 1)" class="ee_img tr_noresize" eeimg="1">  上按均匀分布随机取变量  <img src="https://www.zhihu.com/equation?tex=\xi" alt="\xi" class="ee_img tr_noresize" eeimg="1"> ；
- 求  <img src="https://www.zhihu.com/equation?tex=P^{-1}(\xi)" alt="P^{-1}(\xi)" class="ee_img tr_noresize" eeimg="1"> 

但是这只是一维，通常我们需要在二维上工作。如果  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  相互独立并且其 PDF  <img src="https://www.zhihu.com/equation?tex=p(x, y)" alt="p(x, y)" class="ee_img tr_noresize" eeimg="1">  可以拆分为  <img src="https://www.zhihu.com/equation?tex=p_x(x)p_y(y)" alt="p_x(x)p_y(y)" class="ee_img tr_noresize" eeimg="1"> ，那么分别按照前面一维的方法操作即可。这里注意！拆出的  <img src="https://www.zhihu.com/equation?tex=p_x(x)" alt="p_x(x)" class="ee_img tr_noresize" eeimg="1">  一定要满足  <img src="https://www.zhihu.com/equation?tex=\int p_x(x)\,dx=1" alt="\int p_x(x)\,dx=1" class="ee_img tr_noresize" eeimg="1"> 

不能拆开怎么办？还有招：

- 计算 marginal（边缘）PDF  <img src="https://www.zhihu.com/equation?tex=p(x) = \int{p(x, y)}\,dy" alt="p(x) = \int{p(x, y)}\,dy" class="ee_img tr_noresize" eeimg="1"> ；
- 计算 conditional（条件） PDF  <img src="https://www.zhihu.com/equation?tex=p(y|x) = \frac{p(x, y)}{p(x)}" alt="p(y|x) = \frac{p(x, y)}{p(x)}" class="ee_img tr_noresize" eeimg="1"> ；
- 对这俩函数分别采用一维的方法即可。

回到之前要在单位圆上均匀采样的问题。假设我们要取点  <img src="https://www.zhihu.com/equation?tex=(x, y)" alt="(x, y)" class="ee_img tr_noresize" eeimg="1"> ，易得均匀分布的 PDF  <img src="https://www.zhihu.com/equation?tex=p(x, y) = \frac{1}{\pi}" alt="p(x, y) = \frac{1}{\pi}" class="ee_img tr_noresize" eeimg="1">  。注意这里  <img src="https://www.zhihu.com/equation?tex=x" alt="x" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=y" alt="y" class="ee_img tr_noresize" eeimg="1">  并不是独立的，所以没法按照上面的办法直接拆。我们把  <img src="https://www.zhihu.com/equation?tex=p(x, y)" alt="p(x, y)" class="ee_img tr_noresize" eeimg="1">  转换为更常用的  <img src="https://www.zhihu.com/equation?tex=p(r, \theta)" alt="p(r, \theta)" class="ee_img tr_noresize" eeimg="1"> （其中， <img src="https://www.zhihu.com/equation?tex=r\in[0, 1), \theta\in[0, 2\pi)" alt="r\in[0, 1), \theta\in[0, 2\pi)" class="ee_img tr_noresize" eeimg="1"> ）

## 变换随机变量

我们已经知道了  <img src="https://www.zhihu.com/equation?tex=p(x, y) = \frac{1}{\pi}" alt="p(x, y) = \frac{1}{\pi}" class="ee_img tr_noresize" eeimg="1">  ，那  <img src="https://www.zhihu.com/equation?tex=p(r, \theta)" alt="p(r, \theta)" class="ee_img tr_noresize" eeimg="1">  是多少呢？[wikipedia](https://en.wikipedia.org/wiki/Probability_density_function#Vector_to_vector) 告诉我们，有变换前的变量  <img src="https://www.zhihu.com/equation?tex=x_1" alt="x_1" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=x_2" alt="x_2" class="ee_img tr_noresize" eeimg="1">  的 PDF 为  <img src="https://www.zhihu.com/equation?tex=f" alt="f" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=x_1" alt="x_1" class="ee_img tr_noresize" eeimg="1">  的变换  <img src="https://www.zhihu.com/equation?tex=y_1=H_1(x_1, x_2)" alt="y_1=H_1(x_1, x_2)" class="ee_img tr_noresize" eeimg="1"> ；将  <img src="https://www.zhihu.com/equation?tex=x_2" alt="x_2" class="ee_img tr_noresize" eeimg="1">  的变换  <img src="https://www.zhihu.com/equation?tex=y_2=H_2(x_1, x_2)" alt="y_2=H_2(x_1, x_2)" class="ee_img tr_noresize" eeimg="1"> ； <img src="https://www.zhihu.com/equation?tex=H_1" alt="H_1" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=H_2" alt="H_2" class="ee_img tr_noresize" eeimg="1">  的反函数  <img src="https://www.zhihu.com/equation?tex=x_1 = H_1^{-1}(y_1, y_2)" alt="x_1 = H_1^{-1}(y_1, y_2)" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=x_2 = H_2^{-1}(y_1, y_2)" alt="x_2 = H_2^{-1}(y_1, y_2)" class="ee_img tr_noresize" eeimg="1"> ，那么新的 PDF  <img src="https://www.zhihu.com/equation?tex=g(y_1, y_2)" alt="g(y_1, y_2)" class="ee_img tr_noresize" eeimg="1">  为


<img src="https://www.zhihu.com/equation?tex=g(y_1,y_2) = f_{X_1,X_2}\big(H_1^{-1}(y_1,y_2), H_2^{-1}(y_1,y_2)\big) \left\vert \frac{\partial H_1^{-1}}{\partial y_1} \frac{\partial H_2^{-1}}{\partial y_2} - \frac{\partial H_1^{-1}}{\partial y_2} \frac{\partial H_2^{-1}}{\partial y_1} \right\vert " alt="g(y_1,y_2) = f_{X_1,X_2}\big(H_1^{-1}(y_1,y_2), H_2^{-1}(y_1,y_2)\big) \left\vert \frac{\partial H_1^{-1}}{\partial y_1} \frac{\partial H_2^{-1}}{\partial y_2} - \frac{\partial H_1^{-1}}{\partial y_2} \frac{\partial H_2^{-1}}{\partial y_1} \right\vert " class="ee_img tr_noresize" eeimg="1">


（式子的后半部分为 Jacobian 矩阵的行列式）对于我们这个圆形的例子， <img src="https://www.zhihu.com/equation?tex=x = H_1^{-1}(p, \theta) = r\cos\theta" alt="x = H_1^{-1}(p, \theta) = r\cos\theta" class="ee_img tr_noresize" eeimg="1"> ； <img src="https://www.zhihu.com/equation?tex=y = H_2^{-1}(p, \theta) = r\sin\theta" alt="y = H_2^{-1}(p, \theta) = r\sin\theta" class="ee_img tr_noresize" eeimg="1">  。开开心心求解一下偏微分可以得到


<img src="https://www.zhihu.com/equation?tex=p(r, \theta) = rp(x, y)" alt="p(r, \theta) = rp(x, y)" class="ee_img tr_noresize" eeimg="1">

如果  <img src="https://www.zhihu.com/equation?tex=(x, y)" alt="(x, y)" class="ee_img tr_noresize" eeimg="1">  满足均匀分布，那么可以得到


<img src="https://www.zhihu.com/equation?tex=p(r, \theta) = \frac{r}{\pi}" alt="p(r, \theta) = \frac{r}{\pi}" class="ee_img tr_noresize" eeimg="1">

## 回到圆上采样的问题

 <img src="https://www.zhihu.com/equation?tex=r" alt="r" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1">  是妥妥独立的，我们把  <img src="https://www.zhihu.com/equation?tex=p(r, \theta) = \frac{r}{\pi}" alt="p(r, \theta) = \frac{r}{\pi}" class="ee_img tr_noresize" eeimg="1">  拆一下，注意这里要绞尽脑汁拆成满足  <img src="https://www.zhihu.com/equation?tex=\int_{0}^{1} p_r(r')dr' = 1" alt="\int_{0}^{1} p_r(r')dr' = 1" class="ee_img tr_noresize" eeimg="1"> 。一番思考后拆成  <img src="https://www.zhihu.com/equation?tex=p_r(r) = 2r" alt="p_r(r) = 2r" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=p_{\theta}(\theta) = \frac{1}{2\pi}" alt="p_{\theta}(\theta) = \frac{1}{2\pi}" class="ee_img tr_noresize" eeimg="1"> 。分别求解 CDF：


<img src="https://www.zhihu.com/equation?tex=P_r{r} = \int_{0}^{r} r'\,dr' = r^2" alt="P_r{r} = \int_{0}^{r} r'\,dr' = r^2" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=P_\theta{\theta} = \int_{0}^{\theta} \theta'\,d\theta' = \frac{\theta}{2\pi}" alt="P_\theta{\theta} = \int_{0}^{\theta} \theta'\,d\theta' = \frac{\theta}{2\pi}" class="ee_img tr_noresize" eeimg="1">

对 CDF 求反，把在  <img src="https://www.zhihu.com/equation?tex=[0, 1)" alt="[0, 1)" class="ee_img tr_noresize" eeimg="1">  均匀取的独立变量  <img src="https://www.zhihu.com/equation?tex=\xi_1" alt="\xi_1" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=\xi_2" alt="\xi_2" class="ee_img tr_noresize" eeimg="1">  代入，可得


<img src="https://www.zhihu.com/equation?tex=r = \sqrt{\xi_1}" alt="r = \sqrt{\xi_1}" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=\theta = 2\pi\xi_2" alt="\theta = 2\pi\xi_2" class="ee_img tr_noresize" eeimg="1">


# 立体角的微分

刚才我们说 


<img src="https://www.zhihu.com/equation?tex=p(\omega) \propto \cos\theta" alt="p(\omega) \propto \cos\theta" class="ee_img tr_noresize" eeimg="1">

那么定义我们要求解的常数  <img src="https://www.zhihu.com/equation?tex=c" alt="c" class="ee_img tr_noresize" eeimg="1"> ，则


<img src="https://www.zhihu.com/equation?tex=p(\omega) = c\cos\theta" alt="p(\omega) = c\cos\theta" class="ee_img tr_noresize" eeimg="1">

其 CDF 应满足：


<img src="https://www.zhihu.com/equation?tex=\int_{H^2}c\cos\theta\,d\omega = 1" alt="\int_{H^2}c\cos\theta\,d\omega = 1" class="ee_img tr_noresize" eeimg="1">

这个积分直接求不太好求，我们一般将其转换为对  <img src="https://www.zhihu.com/equation?tex=(\theta, \phi)" alt="(\theta, \phi)" class="ee_img tr_noresize" eeimg="1">  的积分，这里再次借用 PBRT 的图：

![](https://raw.githubusercontent.com/LYP951018/Markdown4Zhihu/master/Data/Sampling/Sin_dtheta_dphi.svg)

其中， <img src="https://www.zhihu.com/equation?tex=\,d\omega" alt="\,d\omega" class="ee_img tr_noresize" eeimg="1">  就是图中灰色部分的面积。在  <img src="https://www.zhihu.com/equation?tex=\,d\theta" alt="\,d\theta" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=\,d\phi" alt="\,d\phi" class="ee_img tr_noresize" eeimg="1">  足够小时，灰色部分是个矩形，求面积只需要乘上俩边长。易见俩边长分别为  <img src="https://www.zhihu.com/equation?tex=\,d\theta" alt="\,d\theta" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=\sin\theta\,d\phi" alt="\sin\theta\,d\phi" class="ee_img tr_noresize" eeimg="1"> （ <img src="https://www.zhihu.com/equation?tex=\sin\theta" alt="\sin\theta" class="ee_img tr_noresize" eeimg="1">  从将球半径投影至平面得到），则


<img src="https://www.zhihu.com/equation?tex=\,d\omega = \sin\theta\,d\theta\,d\phi" alt="\,d\omega = \sin\theta\,d\theta\,d\phi" class="ee_img tr_noresize" eeimg="1">

那么将其代入我们刚才想求的式子  <img src="https://www.zhihu.com/equation?tex=\int_{H^2}c\cos\theta\,d\omega = 1" alt="\int_{H^2}c\cos\theta\,d\omega = 1" class="ee_img tr_noresize" eeimg="1"> ：


<img src="https://www.zhihu.com/equation?tex=\int_0^{2\pi} \int_0^{\frac{\pi}{2}} c\cos\theta\sin\theta\,d\theta\,d\phi = 1" alt="\int_0^{2\pi} \int_0^{\frac{\pi}{2}} c\cos\theta\sin\theta\,d\theta\,d\phi = 1" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=\int_0^{2\pi} \int_0^{\frac{\pi}{2}} \cos\theta\sin\theta\,d\theta\,d\phi = \frac{1}{c}" alt="\int_0^{2\pi} \int_0^{\frac{\pi}{2}} \cos\theta\sin\theta\,d\theta\,d\phi = \frac{1}{c}" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=\int_0^{2\pi}\frac{1}{2}\,d\phi = \frac{1}{c}" alt="\int_0^{2\pi}\frac{1}{2}\,d\phi = \frac{1}{c}" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=c = \frac{1}{\pi}" alt="c = \frac{1}{\pi}" class="ee_img tr_noresize" eeimg="1">

则


<img src="https://www.zhihu.com/equation?tex=p(\theta,\phi) = \frac{\cos\theta\sin\theta}{\pi}" alt="p(\theta,\phi) = \frac{\cos\theta\sin\theta}{\pi}" class="ee_img tr_noresize" eeimg="1">

## Malley’s method 

我们希望在半球表面上取点，使得  <img src="https://www.zhihu.com/equation?tex=p(\theta,\phi) = \frac{\cos\theta\sin\theta}{\pi}" alt="p(\theta,\phi) = \frac{\cos\theta\sin\theta}{\pi}" class="ee_img tr_noresize" eeimg="1"> 。Malley's method 说的是，如果我们能够在圆上均匀取点，将这个点投影到半球表面，那么这样就能满足我们期望的 PDF。

![](https://raw.githubusercontent.com/LYP951018/Markdown4Zhihu/master/Data/Sampling/Malleys_method.svg)

圆上的  <img src="https://www.zhihu.com/equation?tex=(r, \theta)" alt="(r, \theta)" class="ee_img tr_noresize" eeimg="1">  在半球里实际上是  <img src="https://www.zhihu.com/equation?tex=(\sin\theta, \phi)" alt="(\sin\theta, \phi)" class="ee_img tr_noresize" eeimg="1"> ，我们将其变换到  <img src="https://www.zhihu.com/equation?tex=(\theta, \phi)" alt="(\theta, \phi)" class="ee_img tr_noresize" eeimg="1"> ，即


<img src="https://www.zhihu.com/equation?tex=H_1^{-1}(x_1, x_2) = \sin{x_{1}}" alt="H_1^{-1}(x_1, x_2) = \sin{x_{1}}" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=H_2^{-1}(x_1, x_2) = x_2" alt="H_2^{-1}(x_1, x_2) = x_2" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=f_{X_1, X_2} = \frac{X_1}{\pi}" alt="f_{X_1, X_2} = \frac{X_1}{\pi}" class="ee_img tr_noresize" eeimg="1">

求解上面那个大式子，求得球面上的 PDF 为：


<img src="https://www.zhihu.com/equation?tex=g(\theta, \phi) = \frac{\cos\theta\sin\theta}{\pi}" alt="g(\theta, \phi) = \frac{\cos\theta\sin\theta}{\pi}" class="ee_img tr_noresize" eeimg="1">

真巧啊！


## 在三角形中的均匀采样

三角形！渲染中出场率最高的形状。假设该三角形为腰长为 1 的等腰直角三角形，使用重心坐标（barycentric coordinate）来表示要取的点  <img src="https://www.zhihu.com/equation?tex=(u, v)" alt="(u, v)" class="ee_img tr_noresize" eeimg="1"> （其中  <img src="https://www.zhihu.com/equation?tex=u \in (0, 1),v \in (0, 1 - u)" alt="u \in (0, 1),v \in (0, 1 - u)" class="ee_img tr_noresize" eeimg="1"> ），易得  <img src="https://www.zhihu.com/equation?tex=p(u, v) = \frac{1}{S}" alt="p(u, v) = \frac{1}{S}" class="ee_img tr_noresize" eeimg="1"> ，在这里我们有  <img src="https://www.zhihu.com/equation?tex=S = \frac{1}{2}" alt="S = \frac{1}{2}" class="ee_img tr_noresize" eeimg="1"> ，则  <img src="https://www.zhihu.com/equation?tex=p(u, v) = 2" alt="p(u, v) = 2" class="ee_img tr_noresize" eeimg="1"> 。这玩意可不太好拆出来  <img src="https://www.zhihu.com/equation?tex=p_u" alt="p_u" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=p_v" alt="p_v" class="ee_img tr_noresize" eeimg="1"> ，所以我们先求  <img src="https://www.zhihu.com/equation?tex=p(u)" alt="p(u)" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=p(v | u)" alt="p(v | u)" class="ee_img tr_noresize" eeimg="1"> ：


<img src="https://www.zhihu.com/equation?tex=p(u) = \int_{0}^{1 - u}2\,dv = 2(1 - u)" alt="p(u) = \int_{0}^{1 - u}2\,dv = 2(1 - u)" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=p(v | u) = \frac{p(u, v)}{p(u)} = \frac{2}{2(1 - u)} = \frac{1}{1 - u}" alt="p(v | u) = \frac{p(u, v)}{p(u)} = \frac{2}{2(1 - u)} = \frac{1}{1 - u}" class="ee_img tr_noresize" eeimg="1">

然后对二者积分


<img src="https://www.zhihu.com/equation?tex=P_u(u) = \int_{0}^{u}2(1 - u')\,du' = 2u - u^2" alt="P_u(u) = \int_{0}^{u}2(1 - u')\,du' = 2u - u^2" class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=P_v(v) = \int_{0}^{v}p(v'|u) = \int_{0}^{v}\frac{1}{1 - u}\,dv' = \frac{v}{1 - u} " alt="P_v(v) = \int_{0}^{v}p(v'|u) = \int_{0}^{v}\frac{1}{1 - u}\,dv' = \frac{v}{1 - u} " class="ee_img tr_noresize" eeimg="1">

分别求反函数（ <img src="https://www.zhihu.com/equation?tex=P_u" alt="P_u" class="ee_img tr_noresize" eeimg="1">  求解反函数时需要解个一元二次方程，去掉不在  <img src="https://www.zhihu.com/equation?tex=(0, 1)" alt="(0, 1)" class="ee_img tr_noresize" eeimg="1">  范围内的那个解），将  <img src="https://www.zhihu.com/equation?tex=\xi_1" alt="\xi_1" class="ee_img tr_noresize" eeimg="1"> 、 <img src="https://www.zhihu.com/equation?tex=\xi_2" alt="\xi_2" class="ee_img tr_noresize" eeimg="1">  代入：


<img src="https://www.zhihu.com/equation?tex=u = 1 - \sqrt{1 - \xi_1} " alt="u = 1 - \sqrt{1 - \xi_1} " class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=v = \sqrt{1 - \xi_1}\xi_2 " alt="v = \sqrt{1 - \xi_1}\xi_2 " class="ee_img tr_noresize" eeimg="1">

PBRT 多做了一步，它认为可以将  <img src="https://www.zhihu.com/equation?tex=1 - \xi" alt="1 - \xi" class="ee_img tr_noresize" eeimg="1">  替换为  <img src="https://www.zhihu.com/equation?tex=\xi" alt="\xi" class="ee_img tr_noresize" eeimg="1"> ，~~但是我觉得不太严谨：替换了之后定义域会包含原来不包含的  <img src="https://www.zhihu.com/equation?tex=1" alt="1" class="ee_img tr_noresize" eeimg="1"> ，由于  <img src="https://www.zhihu.com/equation?tex=\xi \in [0, 1)" alt="\xi \in [0, 1)" class="ee_img tr_noresize" eeimg="1"> ，替换完  <img src="https://www.zhihu.com/equation?tex=u" alt="u" class="ee_img tr_noresize" eeimg="1">  的范围有微妙的变化~~ 于是最终：


<img src="https://www.zhihu.com/equation?tex=u = 1 - \sqrt{\xi_1} " alt="u = 1 - \sqrt{\xi_1} " class="ee_img tr_noresize" eeimg="1">


<img src="https://www.zhihu.com/equation?tex=v = \sqrt{\xi_1}\xi_2 " alt="v = \sqrt{\xi_1}\xi_2 " class="ee_img tr_noresize" eeimg="1">

上面提到，假设了该三角形为腰长为 1 的等腰直角三角形，换成普通三角形只是  <img src="https://www.zhihu.com/equation?tex=p(u, v)" alt="p(u, v)" class="ee_img tr_noresize" eeimg="1">  不同（依赖三角形的面积  <img src="https://www.zhihu.com/equation?tex=S" alt="S" class="ee_img tr_noresize" eeimg="1"> ），其它都一样。

（注：PBRT 也提到了为什么不使用先在正方形内均匀取点，再把取到的点“折”到三角形的那边。因为那样会导致离的非常远的采样点（例如  <img src="https://www.zhihu.com/equation?tex=(0.01, 0,01)" alt="(0.01, 0,01)" class="ee_img tr_noresize" eeimg="1">  和  <img src="https://www.zhihu.com/equation?tex=(0.99, 0.99)" alt="(0.99, 0.99)" class="ee_img tr_noresize" eeimg="1"> ）映射到一个点上，破环了我们上文提到的 stratified Sampling））