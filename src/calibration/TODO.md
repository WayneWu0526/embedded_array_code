1. 加载位于sensor_data_collection/data/manual_x, manual_y, manual_z的手动标定数据.

2. 加载magnitude.txt文件中的磁场强度数据.

3. 进行数据整理。对于sensor_i，将所有5V的数据（三个channel）进行拼接，同步创建对应voltage的magnitude参考列，为magnitude * ones(N, 1) 的列向量。然后继续拼接4V，直到1V的数据。接着，对所有的测量结果进行椭圆校准和模值计算，得到每一行数据的模值。
最后，得到了Nx1的模值参考值和Nx1的校准值，就可以利用最小二乘计算每个sensor_i的模值增益s_i了

4. R的计算