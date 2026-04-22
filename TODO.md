done 1. 新开分支将一致性校正逻辑改为订阅raw数据，然后内部进行椭圆校正，再进行一致性校正。raw数据计算平均值，和椭圆校正之后的数据进行增益放大。
done 2. tdm可以不再发布ellipsoid中间态结果，直接发布raw和corr结果。

done (本地matlab) 3. 写一个plot，绘制三个tag的结果，绘制可视化 是否能判定完成？

1. 研究当前5cm的精度的原因。

done 1. 将p的计算替换为R, p解耦计算。采用中心化刚体配准

2. 研究近距离和远距离下，同样都是产生10mT磁场，近距离和远距离的X对传感器的需求谁更小。

验证校准后的电磁铁的梯度测量情况。用它和模型结果进行对比。一种潜在可能是 p_C_bar正好位于p_true附近，所以定位效果非常好。（似乎找到了一个作弊办法）

1. 对比梯度

2. 换成简单的计算放大系数S，仿照consistency 单轴测量，利用正负相消的办法

~ 4. 修改stm manual，允许通过topic来保存数据

5. 分离主程序调用calibration

## 系数校正-基于一致性校正的算法

### 系数校正模块的数据获取
1. 建立与一致性校正算法相同的框架，不同的是，系数校正旨在仅给每个传感器标定一个增益系数。
2. 与一致性校正一样，对硬件stm32，fy8300进行初始化
3. 不再需要记录background数据。
5. 将是否开启某个chanel作为launch参数，例如chanel1值为true表示启动chanel1，chanel2,3,false表示他们不启动。将每个chanel的外部施加的理论磁场模值也作为launch参数传入。
6. 开始之前需要终端汇报具体的参数设置，例如chanel1,2,3是否开启，施加的理论磁场模值分别是多少。
6. 采样五十组raw数据，每组都是10次采样的平均值，根据施加的positive/negative来区分csv文件。
7. 采样完毕后，将会仅保存两组csv文件，分别是positive和negative的csv文件，每个csv文件中包含五十行，每行包含三个传感器的平均值。

### 系数校正模块的算法设计
1. 在读取结束后，不结束节点，而是进入postprocessing部分，用于计算增益系数。
2. 由于背景磁场的存在，需要利用positive和negative两组数据进行增益系数的计算。对于每个通道，计算每个传感器在positive和negative两组数据中的平均值，分别记为mean_positive和mean_negative。
3. 将正负两组数据的平均值进行差分，得到每个传感器的差分值diff = mean_positive - mean_negative，这个差分值反映了施加的磁场对传感器的响应，同时也抵消了背景磁场的影响。
4. 计算每个传感器的差分值的模值diff_magnitude = sqrt(diff_x^2 + diff_y^2 + diff_z^2)，这个模值表示了每个传感器对于施加磁场的整体响应程度。
5. 根据施加的理论磁场模值theoretical_magnitude，计算每个传感器的增益系数gain = 2 * theoretical_magnitude / diff_magnitude。这个增益系数表示了每个传感器对于施加磁场的响应程度。
6. 遍历每个开启的chanel，对得到的不同chanel的增益系数进行平均。
7. 将计算得到的所有传感器的增益系数保存到一个csv文件中。