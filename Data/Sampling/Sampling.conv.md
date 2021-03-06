# Ray tracing 中的 sampling

（注：本文基本完全抄袭 [PBRT](http://www.pbr-book.org/3ed-2018/contents.html)）

## 方格子取点

将场景渲染为图像本质上是求解


<img src="https://www.zhihu.com/equation?tex=f(x%2c+y%2c+t%2c+u%2c+v%2c+...)+%5crightarrow+L" alt="[公式]" eeimg="1" data-formula="f(x, y, t, u, v, ...) \rightarrow L">


其中 <img src="https://www.zhihu.com/equation?tex=x" alt="[公式]" eeimg="1" data-formula="x">、<img src="https://www.zhihu.com/equation?tex=y" alt="[公式]" eeimg="1" data-formula="y"> 为 2D 图像中的位置，也是发射出的光线的起点。如果 <img src="https://www.zhihu.com/equation?tex=x" alt="[公式]" eeimg="1" data-formula="x">、<img src="https://www.zhihu.com/equation?tex=y" alt="[公式]" eeimg="1" data-formula="y"> 只取离散的整数点，那么会由于 sampling rate 过低 ~~导致sampling rate < 2 x 最大频率（Nyquist limit），进而~~ 导致 aliasing。要么通过 prefilter 除掉高频，要么提高 sampling rate。我们通常在每个整数像素对应的格子内多取几个点来发射光线（取点的数量叫做 samples per pixel（SPP））来提高 sampling rate。

但由于上文提到的 <img src="https://www.zhihu.com/equation?tex=f" alt="[公式]" eeimg="1" data-formula="f"> 在实际场景中最大频率 <img src="https://www.zhihu.com/equation?tex=%5comega_0" alt="[公式]" eeimg="1" data-formula="\omega_0"> 为 <img src="https://www.zhihu.com/equation?tex=%5cinfty" alt="[公式]" eeimg="1" data-formula="\infty"> （几何体的边界会有一个值的突变），所以即使增大 sampling rate 也不能完全解决 aliasing 的问题。我们一般在取点时使用非均匀采样，使用一些随机因子，而不只是均匀间隔地取（例如 <img src="https://www.zhihu.com/equation?tex=(%5cfrac%7b1%7d%7b4%7d%2c+%5cfrac%7b1%7d%7b4%7d)" alt="[公式]" eeimg="1" data-formula="(\frac{1}{4}, \frac{1}{4})">、<img src="https://www.zhihu.com/equation?tex=(%5cfrac%7b1%7d%7b2%7d%2c+%5cfrac%7b1%7d%7b4%7d)" alt="[公式]" eeimg="1" data-formula="(\frac{1}{2}, \frac{1}{4})"> 这样每隔 <img src="https://www.zhihu.com/equation?tex=%5cfrac%7b1%7d%7b4%7d" alt="[公式]" eeimg="1" data-formula="\frac{1}{4}"> 取点），这样 sampling rate 不再固定，也就会减少固定的 aliasing，但是会产生噪音（noise），据说相比之下观感更好。

借助 [pcg-c](https://github.com/imneme/pcg-c)，我们可以生成 <img src="https://www.zhihu.com/equation?tex=%5b0%2c+1)" alt="[公式]" eeimg="1" data-formula="[0, 1)"> 范围内的随机数 <img src="https://www.zhihu.com/equation?tex=%5cxi" alt="[公式]" eeimg="1" data-formula="\xi">：

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

![](./square_plain.png)

为了使得生成的点覆盖得更广、更均匀，通常在完全随机的基础上使用 stratified Sampling，即预先将形状均匀分成若干个小格子（strata），在每个小格子内再随机采样，如下图所示。

![](./stratified_demo.png)

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

![](./square_stratified.png)

（似乎比刚才更均匀一些（x


## rejection sampling

在正方形中随机取样是挺简单，但是在 ray tracing 中，不仅仅需要在正方形内随机取点。例如使用 Thinlens 相机模型时，需要在 lens （即圆）上随机取点 <img src="https://www.zhihu.com/equation?tex=(u%2c+v)" alt="[公式]" eeimg="1" data-formula="(u, v)">。最朴素的方法是 rejection sampling：我们先调用在正方形里生成随机点的函数 `Sampler::Gen2D`，如果生成的点不在圆内，就再次调用 `Sampler::Gen2D`，直到生成的点在圆内。代码如下：

```cpp
glm::vec2 RejectionSampleDisk(Sampler& sampler) {
    glm::vec2 p;
    do {
        p = 2.0f * sampler.Gen2D() - glm::vec2{ 1.0f, 1.0f };
    } while (glm::length2(p) > 1.0f);
    return p;
}
```

![](./rejection_disk.png)


## Monte Carlo

渲染时往往需要使用积分，例如反射等式：


<img src="https://www.zhihu.com/equation?tex=+L_o(p%2c+%5comega_o)+%3d+%5cint_%7bS%5e2%7d+f(p%2c+%5comega_o%2c+%5comega_i)L_i(p%2c+%5comega_i)+%5ccos%5ctheta_i+%5c%2cd%5comega_i+" alt="[公式]" eeimg="1" data-formula=" L_o(p, \omega_o) = \int_{S^2} f(p, \omega_o, \omega_i)L_i(p, \omega_i) \cos\theta_i \,d\omega_i ">


其中，<img src="https://www.zhihu.com/equation?tex=L_o" alt="[公式]" eeimg="1" data-formula="L_o"> 为在表面上点 <img src="https://www.zhihu.com/equation?tex=p" alt="[公式]" eeimg="1" data-formula="p"> 出射方向为 <img src="https://www.zhihu.com/equation?tex=%5comega_o" alt="[公式]" eeimg="1" data-formula="\omega_o"> 的 exitant radiance；<img src="https://www.zhihu.com/equation?tex=f" alt="[公式]" eeimg="1" data-formula="f"> 为表面的 BRDF；<img src="https://www.zhihu.com/equation?tex=L_i" alt="[公式]" eeimg="1" data-formula="L_i"> 为 incident radiance；<img src="https://www.zhihu.com/equation?tex=%5ctheta_i" alt="[公式]" eeimg="1" data-formula="\theta_i"> 为 <img src="https://www.zhihu.com/equation?tex=%5comega_o" alt="[公式]" eeimg="1" data-formula="\omega_o"> 与表面上点 <img src="https://www.zhihu.com/equation?tex=p" alt="[公式]" eeimg="1" data-formula="p"> 的法线的夹角。

但是计算机求不了这个积分，所以我们使用 Monte Carlo 来随机取几个变量来模拟取积分：


<img src="https://www.zhihu.com/equation?tex=F_N+%3d+%5cfrac%7b1%7d%7bN%7d%5cint_%7bi%3d1%7d%5e%7bN%7d+%5cfrac%7bf(X_i)%7d%7bp(X_i)%7d" alt="[公式]" eeimg="1" data-formula="F_N = \frac{1}{N}\int_{i=1}^{N} \frac{f(X_i)}{p(X_i)}">


其中，随机变量的 <img src="https://www.zhihu.com/equation?tex=X_i" alt="[公式]" eeimg="1" data-formula="X_i"> 的 PDF 为 <img src="https://www.zhihu.com/equation?tex=p(X_i)" alt="[公式]" eeimg="1" data-formula="p(X_i)"> 。<img src="https://www.zhihu.com/equation?tex=p" alt="[公式]" eeimg="1" data-formula="p"> 的形状越和 <img src="https://www.zhihu.com/equation?tex=f" alt="[公式]" eeimg="1" data-formula="f"> 相似，那么 <img src="https://www.zhihu.com/equation?tex=F_N" alt="[公式]" eeimg="1" data-formula="F_N"> 就会越接近要求的积分值。所以我们从 <img src="https://www.zhihu.com/equation?tex=f" alt="[公式]" eeimg="1" data-formula="f"> 找到一个接近其形状的 <img src="https://www.zhihu.com/equation?tex=p" alt="[公式]" eeimg="1" data-formula="p">，然后按照这个 <img src="https://www.zhihu.com/equation?tex=p" alt="[公式]" eeimg="1" data-formula="p"> 来取几个变量 <img src="https://www.zhihu.com/equation?tex=X_i" alt="[公式]" eeimg="1" data-formula="X_i">。就像上面的等式，我们在选取随机的 <img src="https://www.zhihu.com/equation?tex=%5comega_o" alt="[公式]" eeimg="1" data-formula="\omega_o"> 时，可能希望 <img src="https://www.zhihu.com/equation?tex=p(%5comega)" alt="[公式]" eeimg="1" data-formula="p(\omega)"> 满足


<img src="https://www.zhihu.com/equation?tex=p(%5comega)+%5cpropto+%5ccos%5ctheta" alt="[公式]" eeimg="1" data-formula="p(\omega) \propto \cos\theta">


其中，<img src="https://www.zhihu.com/equation?tex=%5comega" alt="[公式]" eeimg="1" data-formula="\omega"> 为选的方向，<img src="https://www.zhihu.com/equation?tex=%5ctheta" alt="[公式]" eeimg="1" data-formula="\theta"> 为表面上的点与球圆心连成的线与 z 轴的夹角。

总而言之，我们希望能够生成 PDF 为指定的 <img src="https://www.zhihu.com/equation?tex=p" alt="[公式]" eeimg="1" data-formula="p"> 的随机变量。


## Inversion method

假设我们想分别按 PDF 为 <img src="https://www.zhihu.com/equation?tex=p_1" alt="[公式]" eeimg="1" data-formula="p_1">、<img src="https://www.zhihu.com/equation?tex=p_2" alt="[公式]" eeimg="1" data-formula="p_2">、<img src="https://www.zhihu.com/equation?tex=p_3" alt="[公式]" eeimg="1" data-formula="p_3">、<img src="https://www.zhihu.com/equation?tex=p_4" alt="[公式]" eeimg="1" data-formula="p_4"> 来采样四个随机变量，我们把它们的 CDF （<img src="https://www.zhihu.com/equation?tex=%5csum_%7b1%7d%5e%7bi%7d+p_i" alt="[公式]" eeimg="1" data-formula="\sum_{1}^{i} p_i">） 绘制出来，如图所示：

![](./discrete-cdf.svg)

如果我们在 <img src="https://www.zhihu.com/equation?tex=%5b0%2c+1)" alt="[公式]" eeimg="1" data-formula="[0, 1)"> 随机取变量 <img src="https://www.zhihu.com/equation?tex=%5cxi" alt="[公式]" eeimg="1" data-formula="\xi">，在 Y 轴画条线找到第一个有交点的变量，就显而易见地满足了指定的 PDF :

![](./discrete-inversion.svg)

将这个方法扩展到连续的随机变量上为：

- 求指定的 PDF <img src="https://www.zhihu.com/equation?tex=p" alt="[公式]" eeimg="1" data-formula="p"> 的 CDF <img src="https://www.zhihu.com/equation?tex=P(x)+%3d+%5cint_%7b0%7d%5e%7bx%7d+p(x%27)%5c%2cdx%27" alt="[公式]" eeimg="1" data-formula="P(x) = \int_{0}^{x} p(x')\,dx'">；
- 求 <img src="https://www.zhihu.com/equation?tex=P%5e%7b-1%7d(x)" alt="[公式]" eeimg="1" data-formula="P^{-1}(x)">；
- 在 <img src="https://www.zhihu.com/equation?tex=%5b0%2c+1)" alt="[公式]" eeimg="1" data-formula="[0, 1)"> 上按均匀分布随机取变量 <img src="https://www.zhihu.com/equation?tex=%5cxi" alt="[公式]" eeimg="1" data-formula="\xi">；
- 求 <img src="https://www.zhihu.com/equation?tex=P%5e%7b-1%7d(%5cxi)" alt="[公式]" eeimg="1" data-formula="P^{-1}(\xi)">

但是这只是一维，通常我们需要在二维上工作。如果 <img src="https://www.zhihu.com/equation?tex=x" alt="[公式]" eeimg="1" data-formula="x">、<img src="https://www.zhihu.com/equation?tex=y" alt="[公式]" eeimg="1" data-formula="y"> 相互独立并且其 PDF <img src="https://www.zhihu.com/equation?tex=p(x%2c+y)" alt="[公式]" eeimg="1" data-formula="p(x, y)"> 可以拆分为 <img src="https://www.zhihu.com/equation?tex=p_x(x)p_y(y)" alt="[公式]" eeimg="1" data-formula="p_x(x)p_y(y)">，那么分别按照前面一维的方法操作即可。这里注意！拆出的 <img src="https://www.zhihu.com/equation?tex=p_x(x)" alt="[公式]" eeimg="1" data-formula="p_x(x)"> 一定要满足 <img src="https://www.zhihu.com/equation?tex=%5cint+p_x(x)%5c%2cdx%3d1" alt="[公式]" eeimg="1" data-formula="\int p_x(x)\,dx=1">

不能拆开怎么办？还有招：

- 计算 marginal（边缘）PDF <img src="https://www.zhihu.com/equation?tex=p(x)+%3d+%5cint%7bp(x%2c+y)%7d%5c%2cdy" alt="[公式]" eeimg="1" data-formula="p(x) = \int{p(x, y)}\,dy">；
- 计算 conditional（条件） PDF <img src="https://www.zhihu.com/equation?tex=p(y%7cx)+%3d+%5cfrac%7bp(x%2c+y)%7d%7bp(x)%7d" alt="[公式]" eeimg="1" data-formula="p(y|x) = \frac{p(x, y)}{p(x)}">；
- 对这俩函数分别采用一维的方法即可。

回到之前要在单位圆上均匀采样的问题。假设我们要取点 <img src="https://www.zhihu.com/equation?tex=(x%2c+y)" alt="[公式]" eeimg="1" data-formula="(x, y)">，易得均匀分布的 PDF <img src="https://www.zhihu.com/equation?tex=p(x%2c+y)+%3d+%5cfrac%7b1%7d%7b%5cpi%7d" alt="[公式]" eeimg="1" data-formula="p(x, y) = \frac{1}{\pi}"> 。注意这里 <img src="https://www.zhihu.com/equation?tex=x" alt="[公式]" eeimg="1" data-formula="x">、<img src="https://www.zhihu.com/equation?tex=y" alt="[公式]" eeimg="1" data-formula="y"> 并不是独立的，所以没法按照上面的办法直接拆。我们把 <img src="https://www.zhihu.com/equation?tex=p(x%2c+y)" alt="[公式]" eeimg="1" data-formula="p(x, y)"> 转换为更常用的 <img src="https://www.zhihu.com/equation?tex=p(r%2c+%5ctheta)" alt="[公式]" eeimg="1" data-formula="p(r, \theta)">（其中，<img src="https://www.zhihu.com/equation?tex=r%5cin%5b0%2c+1)%2c+%5ctheta%5cin%5b0%2c+2%5cpi)" alt="[公式]" eeimg="1" data-formula="r\in[0, 1), \theta\in[0, 2\pi)">）

## 变换随机变量

我们已经知道了 <img src="https://www.zhihu.com/equation?tex=p(x%2c+y)+%3d+%5cfrac%7b1%7d%7b%5cpi%7d" alt="[公式]" eeimg="1" data-formula="p(x, y) = \frac{1}{\pi}"> ，那 <img src="https://www.zhihu.com/equation?tex=p(r%2c+%5ctheta)" alt="[公式]" eeimg="1" data-formula="p(r, \theta)"> 是多少呢？[wikipedia](https://en.wikipedia.org/wiki/Probability_density_function#Vector_to_vector) 告诉我们，有变换前的变量 <img src="https://www.zhihu.com/equation?tex=x_1" alt="[公式]" eeimg="1" data-formula="x_1">、<img src="https://www.zhihu.com/equation?tex=x_2" alt="[公式]" eeimg="1" data-formula="x_2"> 的 PDF 为 <img src="https://www.zhihu.com/equation?tex=f" alt="[公式]" eeimg="1" data-formula="f">，<img src="https://www.zhihu.com/equation?tex=x_1" alt="[公式]" eeimg="1" data-formula="x_1"> 的变换 <img src="https://www.zhihu.com/equation?tex=y_1%3dH_1(x_1%2c+x_2)" alt="[公式]" eeimg="1" data-formula="y_1=H_1(x_1, x_2)">；将 <img src="https://www.zhihu.com/equation?tex=x_2" alt="[公式]" eeimg="1" data-formula="x_2"> 的变换 <img src="https://www.zhihu.com/equation?tex=y_2%3dH_2(x_1%2c+x_2)" alt="[公式]" eeimg="1" data-formula="y_2=H_2(x_1, x_2)">；<img src="https://www.zhihu.com/equation?tex=H_1" alt="[公式]" eeimg="1" data-formula="H_1">、<img src="https://www.zhihu.com/equation?tex=H_2" alt="[公式]" eeimg="1" data-formula="H_2"> 的反函数 <img src="https://www.zhihu.com/equation?tex=x_1+%3d+H_1%5e%7b-1%7d(y_1%2c+y_2)" alt="[公式]" eeimg="1" data-formula="x_1 = H_1^{-1}(y_1, y_2)">、<img src="https://www.zhihu.com/equation?tex=x_2+%3d+H_2%5e%7b-1%7d(y_1%2c+y_2)" alt="[公式]" eeimg="1" data-formula="x_2 = H_2^{-1}(y_1, y_2)">，那么新的 PDF <img src="https://www.zhihu.com/equation?tex=g(y_1%2c+y_2)" alt="[公式]" eeimg="1" data-formula="g(y_1, y_2)"> 为


<img src="https://www.zhihu.com/equation?tex=+g(y_1%2cy_2)+%3d+f_%7bX_1%2cX_2%7d%5cbig(H_1%5e%7b-1%7d(y_1%2cy_2)%2c+H_2%5e%7b-1%7d(y_1%2cy_2)%5cbig)+%5cleft%5cvert+%5cfrac%7b%5cpartial+H_1%5e%7b-1%7d%7d%7b%5cpartial+y_1%7d+%5cfrac%7b%5cpartial+H_2%5e%7b-1%7d%7d%7b%5cpartial+y_2%7d+-+%5cfrac%7b%5cpartial+H_1%5e%7b-1%7d%7d%7b%5cpartial+y_2%7d+%5cfrac%7b%5cpartial+H_2%5e%7b-1%7d%7d%7b%5cpartial+y_1%7d+%5cright%5cvert+" alt="[公式]" eeimg="1" data-formula=" g(y_1,y_2) = f_{X_1,X_2}\big(H_1^{-1}(y_1,y_2), H_2^{-1}(y_1,y_2)\big) \left\vert \frac{\partial H_1^{-1}}{\partial y_1} \frac{\partial H_2^{-1}}{\partial y_2} - \frac{\partial H_1^{-1}}{\partial y_2} \frac{\partial H_2^{-1}}{\partial y_1} \right\vert ">



（式子的后半部分为 Jacobian 矩阵的行列式）对于我们这个圆形的例子，<img src="https://www.zhihu.com/equation?tex=x+%3d+H_1%5e%7b-1%7d(p%2c+%5ctheta)+%3d+r%5ccos%5ctheta" alt="[公式]" eeimg="1" data-formula="x = H_1^{-1}(p, \theta) = r\cos\theta">；<img src="https://www.zhihu.com/equation?tex=y+%3d+H_2%5e%7b-1%7d(p%2c+%5ctheta)+%3d+r%5csin%5ctheta" alt="[公式]" eeimg="1" data-formula="y = H_2^{-1}(p, \theta) = r\sin\theta"> 。开开心心求解一下偏微分可以得到


<img src="https://www.zhihu.com/equation?tex=p(r%2c+%5ctheta)+%3d+rp(x%2c+y)" alt="[公式]" eeimg="1" data-formula="p(r, \theta) = rp(x, y)">


如果 <img src="https://www.zhihu.com/equation?tex=(x%2c+y)" alt="[公式]" eeimg="1" data-formula="(x, y)"> 满足均匀分布，那么可以得到


<img src="https://www.zhihu.com/equation?tex=p(r%2c+%5ctheta)+%3d+%5cfrac%7br%7d%7b%5cpi%7d" alt="[公式]" eeimg="1" data-formula="p(r, \theta) = \frac{r}{\pi}">


## 回到圆上采样的问题

<img src="https://www.zhihu.com/equation?tex=r" alt="[公式]" eeimg="1" data-formula="r">、<img src="https://www.zhihu.com/equation?tex=%5ctheta" alt="[公式]" eeimg="1" data-formula="\theta"> 是妥妥独立的，我们把 <img src="https://www.zhihu.com/equation?tex=p(r%2c+%5ctheta)+%3d+%5cfrac%7br%7d%7b%5cpi%7d" alt="[公式]" eeimg="1" data-formula="p(r, \theta) = \frac{r}{\pi}"> 拆一下，注意这里要绞尽脑汁拆成满足 <img src="https://www.zhihu.com/equation?tex=%5cint_%7b0%7d%5e%7b1%7d+p_r(r%27)dr%27+%3d+1" alt="[公式]" eeimg="1" data-formula="\int_{0}^{1} p_r(r')dr' = 1">。一番思考后拆成 <img src="https://www.zhihu.com/equation?tex=p_r(r)+%3d+2r" alt="[公式]" eeimg="1" data-formula="p_r(r) = 2r">、<img src="https://www.zhihu.com/equation?tex=p_%7b%5ctheta%7d(%5ctheta)+%3d+%5cfrac%7b1%7d%7b2%5cpi%7d" alt="[公式]" eeimg="1" data-formula="p_{\theta}(\theta) = \frac{1}{2\pi}">。分别求解 CDF：


<img src="https://www.zhihu.com/equation?tex=P_r%7br%7d+%3d+%5cint_%7b0%7d%5e%7br%7d+r%27%5c%2cdr%27+%3d+r%5e2" alt="[公式]" eeimg="1" data-formula="P_r{r} = \int_{0}^{r} r'\,dr' = r^2">


<img src="https://www.zhihu.com/equation?tex=P_%5ctheta%7b%5ctheta%7d+%3d+%5cint_%7b0%7d%5e%7b%5ctheta%7d+%5ctheta%27%5c%2cd%5ctheta%27+%3d+%5cfrac%7b%5ctheta%7d%7b2%5cpi%7d" alt="[公式]" eeimg="1" data-formula="P_\theta{\theta} = \int_{0}^{\theta} \theta'\,d\theta' = \frac{\theta}{2\pi}">


对 CDF 求反，把在 <img src="https://www.zhihu.com/equation?tex=%5b0%2c+1)" alt="[公式]" eeimg="1" data-formula="[0, 1)"> 均匀取的独立变量 <img src="https://www.zhihu.com/equation?tex=%5cxi_1" alt="[公式]" eeimg="1" data-formula="\xi_1">、<img src="https://www.zhihu.com/equation?tex=%5cxi_2" alt="[公式]" eeimg="1" data-formula="\xi_2"> 代入，可得


<img src="https://www.zhihu.com/equation?tex=r+%3d+%5csqrt%7b%5cxi_1%7d" alt="[公式]" eeimg="1" data-formula="r = \sqrt{\xi_1}">


<img src="https://www.zhihu.com/equation?tex=%5ctheta+%3d+2%5cpi%5cxi_2" alt="[公式]" eeimg="1" data-formula="\theta = 2\pi\xi_2">



# 立体角的微分

刚才我们说 


<img src="https://www.zhihu.com/equation?tex=p(%5comega)+%5cpropto+%5ccos%5ctheta" alt="[公式]" eeimg="1" data-formula="p(\omega) \propto \cos\theta">


那么定义我们要求解的常数 <img src="https://www.zhihu.com/equation?tex=c" alt="[公式]" eeimg="1" data-formula="c">，则


<img src="https://www.zhihu.com/equation?tex=p(%5comega)+%3d+c%5ccos%5ctheta" alt="[公式]" eeimg="1" data-formula="p(\omega) = c\cos\theta">


其 CDF 应满足：


<img src="https://www.zhihu.com/equation?tex=%5cint_%7bH%5e2%7dc%5ccos%5ctheta%5c%2cd%5comega+%3d+1" alt="[公式]" eeimg="1" data-formula="\int_{H^2}c\cos\theta\,d\omega = 1">


这个积分直接求不太好求，我们一般将其转换为对 <img src="https://www.zhihu.com/equation?tex=(%5ctheta%2c+%5cphi)" alt="[公式]" eeimg="1" data-formula="(\theta, \phi)"> 的积分，这里再次借用 PBRT 的图：

![](./Sin_dtheta_dphi.svg)

其中，<img src="https://www.zhihu.com/equation?tex=%5c%2cd%5comega" alt="[公式]" eeimg="1" data-formula="\,d\omega"> 就是图中灰色部分的面积。在 <img src="https://www.zhihu.com/equation?tex=%5c%2cd%5ctheta" alt="[公式]" eeimg="1" data-formula="\,d\theta">、<img src="https://www.zhihu.com/equation?tex=%5c%2cd%5cphi" alt="[公式]" eeimg="1" data-formula="\,d\phi"> 足够小时，灰色部分是个矩形，求面积只需要乘上俩边长。易见俩边长分别为 <img src="https://www.zhihu.com/equation?tex=%5c%2cd%5ctheta" alt="[公式]" eeimg="1" data-formula="\,d\theta">、<img src="https://www.zhihu.com/equation?tex=%5csin%5ctheta%5c%2cd%5cphi" alt="[公式]" eeimg="1" data-formula="\sin\theta\,d\phi">（<img src="https://www.zhihu.com/equation?tex=%5csin%5ctheta" alt="[公式]" eeimg="1" data-formula="\sin\theta"> 从将球半径投影至平面得到），则


<img src="https://www.zhihu.com/equation?tex=+%5c%2cd%5comega+%3d+%5csin%5ctheta%5c%2cd%5ctheta%5c%2cd%5cphi" alt="[公式]" eeimg="1" data-formula=" \,d\omega = \sin\theta\,d\theta\,d\phi">


那么将其代入我们刚才想求的式子 <img src="https://www.zhihu.com/equation?tex=%5cint_%7bH%5e2%7dc%5ccos%5ctheta%5c%2cd%5comega+%3d+1" alt="[公式]" eeimg="1" data-formula="\int_{H^2}c\cos\theta\,d\omega = 1">：


<img src="https://www.zhihu.com/equation?tex=%5cint_0%5e%7b2%5cpi%7d+%5cint_0%5e%7b%5cfrac%7b%5cpi%7d%7b2%7d%7d+c%5ccos%5ctheta%5csin%5ctheta%5c%2cd%5ctheta%5c%2cd%5cphi+%3d+1" alt="[公式]" eeimg="1" data-formula="\int_0^{2\pi} \int_0^{\frac{\pi}{2}} c\cos\theta\sin\theta\,d\theta\,d\phi = 1">


<img src="https://www.zhihu.com/equation?tex=%5cint_0%5e%7b2%5cpi%7d+%5cint_0%5e%7b%5cfrac%7b%5cpi%7d%7b2%7d%7d+%5ccos%5ctheta%5csin%5ctheta%5c%2cd%5ctheta%5c%2cd%5cphi+%3d+%5cfrac%7b1%7d%7bc%7d" alt="[公式]" eeimg="1" data-formula="\int_0^{2\pi} \int_0^{\frac{\pi}{2}} \cos\theta\sin\theta\,d\theta\,d\phi = \frac{1}{c}">


<img src="https://www.zhihu.com/equation?tex=%5cint_0%5e%7b2%5cpi%7d%5cfrac%7b1%7d%7b2%7d%5c%2cd%5cphi+%3d+%5cfrac%7b1%7d%7bc%7d" alt="[公式]" eeimg="1" data-formula="\int_0^{2\pi}\frac{1}{2}\,d\phi = \frac{1}{c}">


<img src="https://www.zhihu.com/equation?tex=c+%3d+%5cfrac%7b1%7d%7b%5cpi%7d" alt="[公式]" eeimg="1" data-formula="c = \frac{1}{\pi}">


则


<img src="https://www.zhihu.com/equation?tex=p(%5ctheta%2c%5cphi)+%3d+%5cfrac%7b%5ccos%5ctheta%5csin%5ctheta%7d%7b%5cpi%7d" alt="[公式]" eeimg="1" data-formula="p(\theta,\phi) = \frac{\cos\theta\sin\theta}{\pi}">


## Malley’s method 

我们希望在半球表面上取点，使得 <img src="https://www.zhihu.com/equation?tex=p(%5ctheta%2c%5cphi)+%3d+%5cfrac%7b%5ccos%5ctheta%5csin%5ctheta%7d%7b%5cpi%7d" alt="[公式]" eeimg="1" data-formula="p(\theta,\phi) = \frac{\cos\theta\sin\theta}{\pi}">。Malley's method 说的是，如果我们能够在圆上均匀取点，将这个点投影到半球表面，那么这样就能满足我们期望的 PDF。

![](./Malleys_method.svg)

圆上的 <img src="https://www.zhihu.com/equation?tex=(r%2c+%5ctheta)" alt="[公式]" eeimg="1" data-formula="(r, \theta)"> 在半球里实际上是 <img src="https://www.zhihu.com/equation?tex=(%5csin%5ctheta%2c+%5cphi)" alt="[公式]" eeimg="1" data-formula="(\sin\theta, \phi)">，我们将其变换到 <img src="https://www.zhihu.com/equation?tex=(%5ctheta%2c+%5cphi)" alt="[公式]" eeimg="1" data-formula="(\theta, \phi)">，即


<img src="https://www.zhihu.com/equation?tex=H_1%5e%7b-1%7d(x_1%2c+x_2)+%3d+%5csin%7bx_%7b1%7d%7d" alt="[公式]" eeimg="1" data-formula="H_1^{-1}(x_1, x_2) = \sin{x_{1}}">


<img src="https://www.zhihu.com/equation?tex=H_2%5e%7b-1%7d(x_1%2c+x_2)+%3d+x_2" alt="[公式]" eeimg="1" data-formula="H_2^{-1}(x_1, x_2) = x_2">


<img src="https://www.zhihu.com/equation?tex=f_%7bX_1%2c+X_2%7d+%3d+%5cfrac%7bX_1%7d%7b%5cpi%7d" alt="[公式]" eeimg="1" data-formula="f_{X_1, X_2} = \frac{X_1}{\pi}">


求解上面那个大式子，求得球面上的 PDF 为：


<img src="https://www.zhihu.com/equation?tex=g(%5ctheta%2c+%5cphi)+%3d+%5cfrac%7b%5ccos%5ctheta%5csin%5ctheta%7d%7b%5cpi%7d" alt="[公式]" eeimg="1" data-formula="g(\theta, \phi) = \frac{\cos\theta\sin\theta}{\pi}">


真巧啊！


## 在三角形中的均匀采样

三角形！渲染中出场率最高的形状。假设该三角形为腰长为 1 的等腰直角三角形，使用重心坐标（barycentric coordinate）来表示要取的点 <img src="https://www.zhihu.com/equation?tex=(u%2c+v)" alt="[公式]" eeimg="1" data-formula="(u, v)">（其中 <img src="https://www.zhihu.com/equation?tex=u+%5cin+(0%2c+1)%2cv+%5cin+(0%2c+1+-+u)" alt="[公式]" eeimg="1" data-formula="u \in (0, 1),v \in (0, 1 - u)">），易得 <img src="https://www.zhihu.com/equation?tex=p(u%2c+v)+%3d+%5cfrac%7b1%7d%7bS%7d" alt="[公式]" eeimg="1" data-formula="p(u, v) = \frac{1}{S}">，在这里我们有 <img src="https://www.zhihu.com/equation?tex=S+%3d+%5cfrac%7b1%7d%7b2%7d" alt="[公式]" eeimg="1" data-formula="S = \frac{1}{2}">，则 <img src="https://www.zhihu.com/equation?tex=p(u%2c+v)+%3d+2" alt="[公式]" eeimg="1" data-formula="p(u, v) = 2">。这玩意可不太好拆出来 <img src="https://www.zhihu.com/equation?tex=p_u" alt="[公式]" eeimg="1" data-formula="p_u">、<img src="https://www.zhihu.com/equation?tex=p_v" alt="[公式]" eeimg="1" data-formula="p_v">，所以我们先求 <img src="https://www.zhihu.com/equation?tex=p(u)" alt="[公式]" eeimg="1" data-formula="p(u)">、<img src="https://www.zhihu.com/equation?tex=p(v+%7c+u)" alt="[公式]" eeimg="1" data-formula="p(v | u)">：


<img src="https://www.zhihu.com/equation?tex=p(u)+%3d+%5cint_%7b0%7d%5e%7b1+-+u%7d2%5c%2cdv+%3d+2(1+-+u)" alt="[公式]" eeimg="1" data-formula="p(u) = \int_{0}^{1 - u}2\,dv = 2(1 - u)">


<img src="https://www.zhihu.com/equation?tex=p(v+%7c+u)+%3d+%5cfrac%7bp(u%2c+v)%7d%7bp(u)%7d+%3d+%5cfrac%7b2%7d%7b2(1+-+u)%7d+%3d+%5cfrac%7b1%7d%7b1+-+u%7d" alt="[公式]" eeimg="1" data-formula="p(v | u) = \frac{p(u, v)}{p(u)} = \frac{2}{2(1 - u)} = \frac{1}{1 - u}">


然后对二者积分


<img src="https://www.zhihu.com/equation?tex=P_u(u)+%3d+%5cint_%7b0%7d%5e%7bu%7d2(1+-+u%27)%5c%2cdu%27+%3d+2u+-+u%5e2" alt="[公式]" eeimg="1" data-formula="P_u(u) = \int_{0}^{u}2(1 - u')\,du' = 2u - u^2">


<img src="https://www.zhihu.com/equation?tex=P_v(v)+%3d+%5cint_%7b0%7d%5e%7bv%7dp(v%27%7cu)+%3d+%5cint_%7b0%7d%5e%7bv%7d%5cfrac%7b1%7d%7b1+-+u%7d%5c%2cdv%27+%3d+%5cfrac%7bv%7d%7b1+-+u%7d+" alt="[公式]" eeimg="1" data-formula="P_v(v) = \int_{0}^{v}p(v'|u) = \int_{0}^{v}\frac{1}{1 - u}\,dv' = \frac{v}{1 - u} ">


分别求反函数（<img src="https://www.zhihu.com/equation?tex=P_u" alt="[公式]" eeimg="1" data-formula="P_u"> 求解反函数时需要解个一元二次方程，去掉不在 <img src="https://www.zhihu.com/equation?tex=(0%2c+1)" alt="[公式]" eeimg="1" data-formula="(0, 1)"> 范围内的那个解），将 <img src="https://www.zhihu.com/equation?tex=%5cxi_1" alt="[公式]" eeimg="1" data-formula="\xi_1">、<img src="https://www.zhihu.com/equation?tex=%5cxi_2" alt="[公式]" eeimg="1" data-formula="\xi_2"> 代入：


<img src="https://www.zhihu.com/equation?tex=+u+%3d+1+-+%5csqrt%7b1+-+%5cxi_1%7d+" alt="[公式]" eeimg="1" data-formula=" u = 1 - \sqrt{1 - \xi_1} ">


<img src="https://www.zhihu.com/equation?tex=+v+%3d+%5csqrt%7b1+-+%5cxi_1%7d%5cxi_2+" alt="[公式]" eeimg="1" data-formula=" v = \sqrt{1 - \xi_1}\xi_2 ">


PBRT 多做了一步，它认为可以将 <img src="https://www.zhihu.com/equation?tex=1+-+%5cxi" alt="[公式]" eeimg="1" data-formula="1 - \xi"> 替换为 <img src="https://www.zhihu.com/equation?tex=%5cxi" alt="[公式]" eeimg="1" data-formula="\xi">，~~但是我觉得不太严谨：替换了之后定义域会包含原来不包含的 <img src="https://www.zhihu.com/equation?tex=1" alt="[公式]" eeimg="1" data-formula="1">，由于 <img src="https://www.zhihu.com/equation?tex=%5cxi+%5cin+%5b0%2c+1)" alt="[公式]" eeimg="1" data-formula="\xi \in [0, 1)">，替换完 <img src="https://www.zhihu.com/equation?tex=u" alt="[公式]" eeimg="1" data-formula="u"> 的范围有微妙的变化~~ 于是最终：


<img src="https://www.zhihu.com/equation?tex=+u+%3d+1+-+%5csqrt%7b%5cxi_1%7d+" alt="[公式]" eeimg="1" data-formula=" u = 1 - \sqrt{\xi_1} ">


<img src="https://www.zhihu.com/equation?tex=+v+%3d+%5csqrt%7b%5cxi_1%7d%5cxi_2+" alt="[公式]" eeimg="1" data-formula=" v = \sqrt{\xi_1}\xi_2 ">


上面提到，假设了该三角形为腰长为 1 的等腰直角三角形，换成普通三角形只是 <img src="https://www.zhihu.com/equation?tex=p(u%2c+v)" alt="[公式]" eeimg="1" data-formula="p(u, v)"> 不同（依赖三角形的面积 <img src="https://www.zhihu.com/equation?tex=S" alt="[公式]" eeimg="1" data-formula="S">），其它都一样。

（注：PBRT 也提到了为什么不使用先在正方形内均匀取点，再把取到的点“折”到三角形的那边。因为那样会导致离的非常远的采样点（例如 <img src="https://www.zhihu.com/equation?tex=(0.01%2c+0%2c01)" alt="[公式]" eeimg="1" data-formula="(0.01, 0,01)"> 和 <img src="https://www.zhihu.com/equation?tex=(0.99%2c+0.99)" alt="[公式]" eeimg="1" data-formula="(0.99, 0.99)">）映射到一个点上，破环了我们上文提到的 stratified Sampling））