"""
decode_bag.py - 从前视 camera bag 包解码图像帧与时间戳索引

用法:
    python decode_bag.py --config config.yaml
    python decode_bag.py --bag_dir ./data/bags --output_dir ./data/frames
    python decode_bag.py --bag path/to/single.bag --output_dir ./data/frames
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Bag reader 抽象：按优先级尝试 rosbags(ROS2) -> rosbag(ROS1) -> 自定义
# ---------------------------------------------------------------------------

def _try_import_rosbags():
    """尝试导入 rosbags (ROS2)."""
    try:
        from rosbags.rosbag2 import Reader as Ros2Reader
        from rosbags.serde import deserialize_cdr
        return Ros2Reader, deserialize_cdr
    except ImportError:
        return None, None


def _try_import_rosbag():
    """尝试导入 rosbag (ROS1)."""
    try:
        import rosbag
        return rosbag
    except ImportError:
        return None


def list_topics_ros1(bag_path):
    """列出 ROS1 bag 中所有 topic."""
    rosbag = _try_import_rosbag()
    if rosbag is None:
        return []
    bag = rosbag.Bag(str(bag_path), "r")
    topics = list(bag.get_type_and_topic_info().topics.keys())
    bag.close()
    return topics


def list_topics_ros2(bag_path):
    """列出 ROS2 bag 中所有 topic."""
    Ros2Reader, _ = _try_import_rosbags()
    if Ros2Reader is None:
        return []
    with Ros2Reader(bag_path) as reader:
        return [c.topic for c in reader.connections]


def find_front_topic(available_topics, candidate_topics):
    """从候选列表中匹配前视 camera topic."""
    for candidate in candidate_topics:
        for topic in available_topics:
            if candidate in topic:
                return topic
    # 回退：找包含 front 和 image 的 topic
    for topic in available_topics:
        t = topic.lower()
        if "front" in t and ("image" in t or "camera" in t):
            return topic
    return None


def decode_image_msg(msg, msg_type=None):
    """从 ROS 消息解码图像，支持 raw 和 compressed."""
    if msg_type and "Compressed" in msg_type:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    # sensor_msgs/Image raw
    if hasattr(msg, "encoding") and hasattr(msg, "data"):
        encoding = msg.encoding
        h, w = msg.height, msg.width
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if encoding in ("rgb8", "bgr8"):
            img = buf.reshape(h, w, 3)
            if encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        elif encoding in ("mono8",):
            return buf.reshape(h, w)
        elif encoding in ("bayer_rggb8", "bayer_gbrg8", "bayer_grbg8", "bayer_bggr8"):
            code_map = {
                "bayer_rggb8": cv2.COLOR_BayerBG2BGR,
                "bayer_gbrg8": cv2.COLOR_BayerGR2BGR,
                "bayer_grbg8": cv2.COLOR_BayerGB2BGR,
                "bayer_bggr8": cv2.COLOR_BayerRG2BGR,
            }
            img = buf.reshape(h, w)
            return cv2.cvtColor(img, code_map.get(encoding, cv2.COLOR_BayerBG2BGR))
    # 尝试当 compressed 处理
    buf = np.frombuffer(msg.data if hasattr(msg, "data") else msg, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


# ---------------------------------------------------------------------------
# ROS1 解码
# ---------------------------------------------------------------------------

def decode_bag_ros1(bag_path, topic, output_dir, fmt="jpg", quality=95, max_frames=None):
    """从 ROS1 bag 导出图像帧."""
    rosbag = _try_import_rosbag()
    if rosbag is None:
        raise ImportError("需要安装 rosbag: pip install rosbag")
    bag = rosbag.Bag(str(bag_path), "r")
    msg_count = bag.get_message_count(topic)
    if max_frames:
        msg_count = min(msg_count, max_frames)

    os.makedirs(output_dir, exist_ok=True)
    index_rows = []
    frame_idx = 0

    for _, msg, t in tqdm(bag.read_messages(topics=[topic]),
                          total=msg_count, desc=f"ROS1 {Path(bag_path).name}"):
        if max_frames and frame_idx >= max_frames:
            break
        img = decode_image_msg(msg, topic)
        if img is None:
            continue
        ts = t.to_sec() if hasattr(t, "to_sec") else float(t)
        fname = f"{frame_idx:06d}.{fmt}"
        fpath = os.path.join(output_dir, fname)
        if fmt == "jpg":
            cv2.imwrite(fpath, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(fpath, img)
        index_rows.append({"frame_idx": frame_idx, "timestamp": ts, "filename": fname})
        frame_idx += 1

    bag.close()
    return index_rows


# ---------------------------------------------------------------------------
# ROS2 解码
# ---------------------------------------------------------------------------

def decode_bag_ros2(bag_path, topic, output_dir, fmt="jpg", quality=95, max_frames=None):
    """从 ROS2 bag 导出图像帧."""
    Ros2Reader, deserialize_cdr = _try_import_rosbags()
    if Ros2Reader is None:
        raise ImportError("需要安装 rosbags: pip install rosbags")

    os.makedirs(output_dir, exist_ok=True)
    index_rows = []
    frame_idx = 0

    with Ros2Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise ValueError(f"Topic {topic} not found in bag")
        conn = connections[0]
        for _, timestamp, rawdata in tqdm(reader.messages(connections=[conn]),
                                          desc=f"ROS2 {Path(bag_path).name}"):
            if max_frames and frame_idx >= max_frames:
                break
            msg = deserialize_cdr(rawdata, conn.msgtype)
            img = decode_image_msg(msg, conn.msgtype)
            if img is None:
                continue
            ts = timestamp / 1e9  # nanoseconds -> seconds
            fname = f"{frame_idx:06d}.{fmt}"
            fpath = os.path.join(output_dir, fname)
            if fmt == "jpg":
                cv2.imwrite(fpath, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(fpath, img)
            index_rows.append({"frame_idx": frame_idx, "timestamp": ts, "filename": fname})
            frame_idx += 1

    return index_rows


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def detect_bag_type(bag_path):
    """检测 bag 类型: ros1 / ros2 / unknown."""
    bag_path = Path(bag_path)
    if bag_path.is_dir():
        # ROS2 bag 通常是目录
        if any(bag_path.glob("*.db3")):
            return "ros2"
    if bag_path.suffix == ".bag":
        return "ros1"
    if bag_path.suffix == ".db3":
        return "ros2"
    return "unknown"


def decode_single_bag(bag_path, output_dir, cfg):
    """解码单个 bag，自动检测格式，返回帧索引."""
    bag_path = Path(bag_path)
    bag_type = detect_bag_type(bag_path)
    candidate_topics = cfg["decode"]["front_camera_topics"]
    fmt = cfg["decode"].get("output_format", "jpg")
    quality = cfg["decode"].get("jpeg_quality", 95)
    max_frames = cfg["decode"].get("max_frames_per_bag")

    # 列出 topic 并匹配前视
    if bag_type == "ros1":
        topics = list_topics_ros1(bag_path)
    elif bag_type == "ros2":
        topics = list_topics_ros2(bag_path)
    else:
        print(f"[WARN] 无法识别 bag 类型: {bag_path}, 尝试 ROS1")
        topics = list_topics_ros1(bag_path)
        if not topics:
            topics = list_topics_ros2(bag_path)
            bag_type = "ros2"
        else:
            bag_type = "ros1"

    if not topics:
        print(f"[ERROR] 无法读取 {bag_path} 的 topic 列表")
        return []

    front_topic = find_front_topic(topics, candidate_topics)
    if front_topic is None:
        print(f"[ERROR] 未找到前视 topic, 可用 topics: {topics}")
        return []

    print(f"  bag类型: {bag_type}, 前视topic: {front_topic}")

    bag_name = bag_path.stem
    clip_output_dir = os.path.join(output_dir, bag_name)

    if bag_type == "ros1":
        rows = decode_bag_ros1(bag_path, front_topic, clip_output_dir, fmt, quality, max_frames)
    else:
        rows = decode_bag_ros2(bag_path, front_topic, clip_output_dir, fmt, quality, max_frames)

    # 保存时间戳索引
    if rows:
        index_path = os.path.join(clip_output_dir, "timestamp_index.csv")
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_idx", "timestamp", "filename"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  导出 {len(rows)} 帧 -> {clip_output_dir}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Bag 解码：导出前视 camera 图像帧")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--bag", default=None, help="单个 bag 路径（覆盖 config 中的 bag_dir）")
    parser.add_argument("--bag_dir", default=None, help="bag 目录（覆盖 config）")
    parser.add_argument("--output_dir", default=None, help="输出目录（覆盖 config）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bag_dir = args.bag_dir or cfg["data"]["bag_dir"]
    output_dir = args.output_dir or cfg["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 收集 bag 文件
    if args.bag:
        bag_files = [Path(args.bag)]
    else:
        bag_dir = Path(bag_dir)
        bag_files = sorted(list(bag_dir.glob("*.bag")) + list(bag_dir.glob("*.db3")))
        # 也检查子目录（ROS2 bag 可能是目录形式）
        for d in sorted(bag_dir.iterdir()):
            if d.is_dir() and any(d.glob("*.db3")):
                bag_files.append(d)

    if not bag_files:
        print(f"[ERROR] 在 {bag_dir} 中未找到 bag 文件")
        sys.exit(1)

    print(f"找到 {len(bag_files)} 个 bag 文件")
    summary = []

    for bag_path in bag_files:
        print(f"\n处理: {bag_path}")
        rows = decode_single_bag(bag_path, output_dir, cfg)
        summary.append({
            "bag": str(bag_path.name),
            "frames": len(rows),
            "status": "ok" if rows else "failed"
        })

    # 输出汇总
    print("\n===== 解码汇总 =====")
    for s in summary:
        print(f"  {s['bag']}: {s['frames']} 帧 [{s['status']}]")

    ok_count = sum(1 for s in summary if s["status"] == "ok")
    print(f"\n成功: {ok_count}/{len(summary)}")


if __name__ == "__main__":
    main()
