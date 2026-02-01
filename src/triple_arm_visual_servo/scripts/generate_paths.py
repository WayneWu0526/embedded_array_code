#!/usr/bin/env python3
import argparse
import json
import math
import os
# 到时候生成两个圆环，一个点，一个螺旋线，还有一个8

def _normalize(v):
    n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if n < 1e-12:
        return (0.0, 0.0, 0.0), 0.0
    return (v[0]/n, v[1]/n, v[2]/n), n

def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def dist2ori(focus, position, forward_axis="z"):
	# 把原有的前向轴forward_axis旋转到 focus 的方向

	# 方向：背对 focus（向外）
	if position["z"] - focus["z"] > 0:
		dir_vec = (position["x"] - focus["x"],
				   position["y"] - focus["y"],
				   position["z"] - focus["z"])
	else:
		dir_vec = (focus["x"] - position["x"],
				   focus["y"] - position["y"],
				   focus["z"] - position["z"])
	# y_norm = (0.0, 1.0, 0.0)
	# dir_vec = _cross(y_norm, dir_vec)
	b, norm = _normalize(dir_vec)
	if norm == 0.0:
		return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

	# 选择“末端前向轴” a：你希望这根轴对准 b
	if forward_axis == "x":
		a = (1.0, 0.0, 0.0)
	elif forward_axis == "y":
		a = (0.0, 1.0, 0.0)
	else:  # "z"
		a = (0.0, 0.0, 1.0)

	c = _cross(a, b)
	d = _dot(a, b)

	# 同向：不需要旋转
	if d > 0.999999:
		return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

	# 反向：180°，旋转轴不唯一，选一个与 a 不平行的轴来稳定构造
	if d < -0.999999:
		t = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
		axis = _cross(a, t)
		axis, _ = _normalize(axis)
		return {"x": axis[0], "y": axis[1], "z": axis[2], "w": 0.0}

	# 一般情况：q = [1+d, cross(a,b)] 再归一化
	qw = 1.0 + d
	qx, qy, qz = c
	qn = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
	return {"x": qx/qn, "y": qy/qn, "z": qz/qn, "w": qw/qn}


def generate_ring(center_x: float, center_y: float, center_z: float,
				  radius: float, points: int, frame_id: str):
	# 生成圆环轨迹，姿态为单位四元数
	poses = []
	focus = {"x": center_x, "y": center_y, "z": center_z+0.25}
	for i in range(points):
		theta = 2.0 * math.pi * (i / points)
		x = center_x + radius * math.sin(theta)
		y = center_y + radius * math.cos(theta)
		z = center_z 
		# position = {"x": x, "y": y, "z": z}
		position_o = {"x": x, "y": y, "z": z}
		orientation = dist2ori(focus, position_o)
		poses.append({
			"position": {"x": x, "y": y, "z": z},
			# "position": {"x": center_x, "y":center_y, "z":center_z},
			"orientation": orientation
		})

	return {
		"frame_id": frame_id,
		"type": "ring",
		"scale": 1.0,
		"points": points,
		"radius": radius,
		"center": {"x": center_x, "y": center_y, "z": center_z},
		"poses": poses
	}


def generate_spiral(center_x: float, center_y: float, center_z: float,
					 radius: float, height: float, turns: float, points: int, frame_id: str):
	# 生成螺旋线轨迹，姿态为单位四元数
	poses = []
	focus = {"x": center_x, "y": center_y, "z": center_z + height + 0.25}
	for i in range(points):
		# t: 0 -> 1
		t = i / max(points - 1, 1)
		theta = 2.0 * math.pi * turns * t
		x = center_x + radius * math.sin(theta)
		y = center_y + radius * math.cos(theta)
		z = center_z + height * t
		position_o = {"x": x, "y": y, "z": z}
		orientation = dist2ori(focus, position_o)
		poses.append({
			"position": {"x": x, "y": y, "z": z},
			"orientation": orientation
		})

	return {
		"frame_id": frame_id,
		"type": "spiral",
		"scale": 1.0,
		"points": points,
		"radius": radius,
		"height": height,
		"turns": turns,
		"center": {"x": center_x, "y": center_y, "z": center_z},
		"poses": poses,
	}


def generate_spiral_plane(center_x: float, center_y: float, center_z: float,
						 radius_start: float, radius_end: float, turns: float, points: int, frame_id: str):
	# 生成z轴平面固定的漩涡线轨迹，姿态为单位四元数
	poses = []
	focus = {"x": center_x, "y": center_y, "z": center_z + 0.25}
	for i in range(points):
		t = i / max(points - 1, 1)
		theta = 2.0 * math.pi * turns * t
		r = radius_start + (radius_end - radius_start) * t
		x = center_x + r * math.sin(theta)
		y = center_y + r * math.cos(theta)
		z = center_z
		position_o = {"x": x, "y": y, "z": z}
		orientation = dist2ori(focus, position_o)
		poses.append({
			"position": {"x": x, "y": y, "z": z},
			"orientation": orientation
		})

	return {
		"frame_id": frame_id,
		"type": "spiral_plane",
		"scale": 1.0,
		"points": points,
		"radius_start": radius_start,
		"radius_end": radius_end,
		"turns": turns,
		"center": {"x": center_x, "y": center_y, "z": center_z},
		"poses": poses
	}


def generate_lissajous(center_x: float, center_y: float, center_z: float,
					  amp_x: float, amp_y: float, amp_z: float,
					  freq_x: float, freq_y: float, freq_z: float,
					  phase_x: float, phase_y: float, phase_z: float,
					  points: int, frame_id: str):
	# 生成李萨如轨迹，姿态固定为单位四元数
	poses = []
	for i in range(points):
		t = 2.0 * math.pi * (i / max(points, 1))
		x = center_x + amp_x * math.sin(freq_x * t + phase_x)
		y = center_y + amp_y * math.sin(freq_y * t + phase_y)
		z = center_z + amp_z * math.sin(freq_z * t + phase_z)
		poses.append({
			"position": {"x": x, "y": y, "z": z},
			"orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
		})

	return {
		"frame_id": frame_id,
		"type": "lissajous",
		"scale": 1.0,
		"points": points,
		"amplitude": {"x": amp_x, "y": amp_y, "z": amp_z},
		"frequency": {"x": freq_x, "y": freq_y, "z": freq_z},
		"phase": {"x": phase_x, "y": phase_y, "z": phase_z},
		"center": {"x": center_x, "y": center_y, "z": center_z},
		"poses": poses
	}



def main():
	parser = argparse.ArgumentParser(description="Generate a ring trajectory with N points.")
	parser.add_argument("--type", type=str, default="lissajous", choices=["ring", "spiral", "spiral_plane", "lissajous"],
					help="Trajectory type: ring or spiral")
	parser.add_argument("--center-x", type=float, default=0.00, help="Center X (meters)")
	parser.add_argument("--center-y", type=float, default=0.00, help="Center Y (meters)")
	parser.add_argument("--center-z", type=float, default=0.00, help="Center Z (meters)")
	parser.add_argument("--radius", type=float, default=0.16, help="Ring radius (meters)")
	parser.add_argument("--radius-start", type=float, default=0.0, help="Spiral-plane start radius (meters)")
	parser.add_argument("--radius-end", type=float, default=None, help="Spiral-plane end radius (meters)")
	parser.add_argument("--height", type=float, default=0.12, help="Spiral height (meters)")
	parser.add_argument("--turns", type=float, default=2.0, help="Spiral turns")
	parser.add_argument("--points", type=int, default=60, help="Number of points on the ring")
	parser.add_argument("--amp-x", type=float, default=0.16, help="Lissajous amplitude X (meters)")
	parser.add_argument("--amp-y", type=float, default=0.16, help="Lissajous amplitude Y (meters)")
	parser.add_argument("--amp-z", type=float, default=0.16, help="Lissajous amplitude Z (meters)")
	parser.add_argument("--freq-x", type=float, default=3.0, help="Lissajous frequency X")
	parser.add_argument("--freq-y", type=float, default=2.0, help="Lissajous frequency Y")
	parser.add_argument("--freq-z", type=float, default=1.0, help="Lissajous frequency Z")
	parser.add_argument("--phase-x", type=float, default=0.0, help="Lissajous phase X (radians)")
	parser.add_argument("--phase-y", type=float, default=0.0, help="Lissajous phase Y (radians)")
	parser.add_argument("--phase-z", type=float, default=0.0, help="Lissajous phase Z (radians)")
	parser.add_argument("--frame-id", type=str, default="world", help="TF frame id for the trajectory")
	parser.add_argument("--output", type=str, default=None, help="Output JSON file path")

	args = parser.parse_args()

	if args.type == "spiral":
		traj = generate_spiral(
			center_x=args.center_x,
			center_y=args.center_y,
			center_z=args.center_z,
			radius=args.radius,
			height=args.height,
			turns=args.turns,
			points=args.points,
			frame_id=args.frame_id,
		)
	elif args.type == "lissajous":
		traj = generate_lissajous(
			center_x=args.center_x,
			center_y=args.center_y,
			center_z=args.center_z,
			amp_x=args.amp_x,
			amp_y=args.amp_y,
			amp_z=args.amp_z,
			freq_x=args.freq_x,
			freq_y=args.freq_y,
			freq_z=args.freq_z,
			phase_x=args.phase_x,
			phase_y=args.phase_y,
			phase_z=args.phase_z,
			points=args.points,
			frame_id=args.frame_id,
		)
	elif args.type == "spiral_plane":
		radius_end = args.radius if args.radius_end is None else args.radius_end
		traj = generate_spiral_plane(
			center_x=args.center_x,
			center_y=args.center_y,
			center_z=args.center_z,
			radius_start=args.radius_start,
			radius_end=radius_end,
			turns=args.turns,
			points=args.points,
			frame_id=args.frame_id,
		)
	else:
		traj = generate_ring(
			center_x=args.center_x,
			center_y=args.center_y,
			center_z=args.center_z,
			radius=args.radius,
			points=args.points,
			frame_id=args.frame_id,
		)

	# 默认输出路径：自动命名
	base_dir = "D:/proj/gen_path/paths"
	if args.type == "spiral":
		file_name = f"{args.type}_p{args.points}.json"
	elif args.type == "lissajous":
		file_name = (
			f"{args.type}_p{args.points}.json"
			# f"_fx{args.freq_x:g}_fy{args.freq_y:g}_fz{args.freq_z:g}"
			# f"_ax{args.amp_x:.3f}_ay{args.amp_y:.3f}_az{args.amp_z:.3f}.json"
		)
	elif args.type == "spiral_plane":
		radius_end = args.radius if args.radius_end is None else args.radius_end
		file_name = (
			f"{args.type}_p{args.points}.json"
		)
	else:
		file_name = f"{args.type}_p{args.points}_r{args.radius:.3f}.json"
	default_output = os.path.join(base_dir, file_name)
	output_path = args.output or default_output

	# 确保目录存在
	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(traj, f, ensure_ascii=False, indent=2)

	print(f"Trajectory saved to: {output_path}")


if __name__ == "__main__":
	main()

