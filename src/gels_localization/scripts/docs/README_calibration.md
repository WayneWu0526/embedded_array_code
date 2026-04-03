# calibration

## 说明

calibration流程：

1. 从json中加载cycle，其中包含三个sources的位姿，真实Rp，以及传感器raw数据
2. 对每一个slot，完成以下分析：
    1. 将raw数据经过R_CORR和单位换算，得到sensor坐标系下的传感器数据
    2. 对每一个slot，计算经过传感器数据估计出的局部梯度张量和局部磁力
    3. 将估计出的X和b，结合真实坐标R，p，计算slot对应sources的position:p_Ci
    具体公式为：p_Ci_est = p + 3 * R.T @ (X.inverse() @ b)
    4. 对比三个slot计算的p_Ci_est和真实的p_Ci，并进行误差分析

上述目的是：判断当前的R_CORR、单位换算、传感器采样、局部梯度和磁力估计是否正确，判断p_Ci_est和真实p_Ci的误差是否在合理范围内。

## 使用方法

```bash
# 进入 scripts 目录
cd src/gels_localization/scripts/

# 分析所有 cycle 的校准误差
python sensor_calibration.py /path/to/result/

# 分析特定 cycle ID（例如 cycle_0005.json）
python sensor_calibration.py --id=5 /path/to/result/

# 直接指定单个文件
python sensor_calibration.py /path/to/cycle_0000.json
```