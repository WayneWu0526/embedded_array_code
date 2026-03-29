uint8 mode        # 0x01=CVT, 0x02=CCI
uint16 bitmap     # bit0=sensor1, bit11=sensor12
uint16 settling_time  # 单位 0.01ms
uint32 cycle_time     # 单位 0.01ms, 最大 10000.00ms
uint8 cycle_num       # 总 cycle 数, 1 byte
