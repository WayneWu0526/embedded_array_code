#!/usr/bin/env python3
"""格式化打印 stm_magnitude_raw 和 stm_magnitude 对比"""
import rospy
from std_msgs.msg import Float32MultiArray

latest_magnitude = None
latest_magnitude_raw = None
last_print_time = None
print_interval = 0.2  # 最小打印间隔（秒），默认 5 Hz，避免终端刷屏
value_width = 5
value_precision = 2
raw_rate_state = {"count": 0, "window_start": None, "hz": 0.0}
corr_rate_state = {"count": 0, "window_start": None, "hz": 0.0}


def update_rate(state):
    now = rospy.Time.now()
    if state["window_start"] is None:
        state["window_start"] = now
        state["count"] = 1
        return

    state["count"] += 1
    elapsed = (now - state["window_start"]).to_sec()
    if elapsed >= 1.0:
        state["hz"] = state["count"] / elapsed
        state["count"] = 0
        state["window_start"] = now


def cb_magnitude(msg):
    global latest_magnitude
    latest_magnitude = msg.data
    update_rate(corr_rate_state)
    print_comparison()

def cb_magnitude_raw(msg):
    global latest_magnitude_raw
    latest_magnitude_raw = msg.data
    update_rate(raw_rate_state)


def format_values(values):
    return " ".join(f"{v:{value_width}.{value_precision}f}" for v in values)


def format_line(label, hz, values):
    return f"{label:<4} {hz:7.1f} Hz | {format_values(values)}"


def print_comparison():
    global latest_magnitude, latest_magnitude_raw, last_print_time

    # 延迟初始化 last_print_time
    if last_print_time is None:
        last_print_time = rospy.Time.now()
        return  # 第一次直接返回，不打印

    # 时间间隔限制
    now = rospy.Time.now()
    if (now - last_print_time).to_sec() < print_interval:
        return
    last_print_time = now

    # 构造输出行
    output = []

    if latest_magnitude_raw is not None:
        output.append(format_line("raw", raw_rate_state["hz"], latest_magnitude_raw))

    if latest_magnitude is not None:
        output.append(format_line("corr", corr_rate_state["hz"], latest_magnitude))

    # 打印非空行
    for line in output:
        rospy.loginfo(line)

if __name__ == '__main__':
    rospy.init_node('show_magnitude')
    print_interval = float(rospy.get_param('~print_interval', print_interval))
    value_width = int(rospy.get_param('~value_width', value_width))
    value_precision = int(rospy.get_param('~value_precision', value_precision))
    topic_corr = rospy.get_param('~topic_corr', 'stm_magnitude')
    topic_raw = rospy.get_param('~topic_raw', 'stm_magnitude_raw')
    rospy.Subscriber(topic_corr, Float32MultiArray, cb_magnitude)
    rospy.Subscriber(topic_raw, Float32MultiArray, cb_magnitude_raw)

    rospy.loginfo(
        f"Subscribed to {topic_corr} and {topic_raw}; "
        f"print_interval={print_interval:.3f}s, "
        f"value_width={value_width}, value_precision={value_precision}"
    )
    rospy.spin()
