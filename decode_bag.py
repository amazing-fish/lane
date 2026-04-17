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
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import deque
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
    # 回退：找常见前视/相机/视频 topic（例如 /cam_1）
    for topic in available_topics:
        t = topic.lower()
        if any(k in t for k in ("front", "camera", "cam", "video", "image")):
            return topic
    return None


def decode_image_msg(msg, msg_type=None):
    """从 ROS 消息解码图像，支持 raw 和 compressed."""
    try:
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
        raw = msg.data if hasattr(msg, "data") else msg
        if not isinstance(raw, (bytes, bytearray, memoryview, np.ndarray, list, tuple)):
            return None
        buf = np.frombuffer(bytes(raw), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _to_bytes(data):
    """将消息字段统一转换为 bytes."""
    if data is None:
        return None
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, np.ndarray):
        return data.astype(np.uint8).tobytes()
    if isinstance(data, (list, tuple)):
        try:
            return bytes(data)
        except ValueError:
            return None
    return None


def extract_packet_payload(msg):
    """从可能的视频包消息中提取 payload bytes."""
    for field in ("raw_data", "payload", "data"):
        if hasattr(msg, field):
            payload = _to_bytes(getattr(msg, field))
            if payload:
                return payload
    return None


class H265PacketDecoder:
    """基于 ffmpeg 的 H.265 包解码器（单包 + 上下文拼接回退）."""

    def __init__(self, context_packets=24, ffmpeg_threads=0, ffmpeg_hwaccel="auto"):
        self.context_packets = max(1, int(context_packets))
        self.packet_buffer = deque(maxlen=self.context_packets)
        self.ffmpeg_available = shutil.which("ffmpeg") is not None
        self.ffmpeg_threads = max(0, int(ffmpeg_threads))
        self.ffmpeg_hwaccel = (ffmpeg_hwaccel or "").strip().lower()
        self.hwaccel_enabled = self.ffmpeg_hwaccel not in ("", "none", "off", "cpu")
        self.hwaccel_ready = self.ffmpeg_available and self.hwaccel_enabled

    @staticmethod
    def _looks_like_annexb(packet_bytes):
        return (
            len(packet_bytes) > 4
            and packet_bytes[0] == 0x00
            and packet_bytes[1] == 0x00
            and packet_bytes[2] in (0x00, 0x01)
        )

    def _decode_stream(self, stream_bytes):
        if not self.ffmpeg_available or not stream_bytes:
            return None

        base_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if self.ffmpeg_threads > 0:
            base_cmd.extend(["-threads", str(self.ffmpeg_threads)])
        out_cmd = [
            "-f",
            "hevc",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        cmd = list(base_cmd)
        if self.hwaccel_ready:
            cmd.extend(["-hwaccel", self.ffmpeg_hwaccel])
        cmd.extend(out_cmd)
        try:
            proc = subprocess.run(
                cmd,
                input=stream_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0 or not proc.stdout:
            if self.hwaccel_ready:
                self.hwaccel_ready = False
                return self._decode_stream(stream_bytes)
            return None
        arr = np.frombuffer(proc.stdout, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def decode_packet(self, packet_bytes):
        if not packet_bytes:
            return None
        self.packet_buffer.append(packet_bytes)
        img = self._decode_stream(packet_bytes)
        if img is not None:
            return img
        merged = b"".join(self.packet_buffer)
        return self._decode_stream(merged)


def maybe_decode_packet_message(msg, msg_type, packet_decoder):
    """尝试解码视频包消息，返回图像或 None."""
    payload = extract_packet_payload(msg)
    if not payload:
        return None

    video_format = str(getattr(msg, "video_format", "")).lower()
    msg_type_l = (msg_type or "").lower()
    has_raw_image_meta = all(hasattr(msg, k) for k in ("encoding", "height", "width"))
    is_h265_declared = any(k in video_format for k in ("h265", "hevc")) or any(
        k in msg_type_l for k in ("h265", "hevc")
    )

    # 原生 raw image 不走 packet 解码；标准 compressed 路径优先由 decode_image_msg 处理
    if has_raw_image_meta:
        return None
    if not is_h265_declared and not hasattr(msg, "raw_data"):
        # 非明确 H.265 消息，仅在 payload 看起来像 AnnexB 时再尝试
        if not packet_decoder._looks_like_annexb(payload):
            return None
    return packet_decoder.decode_packet(payload)


class AsyncFrameWriter:
    """异步写盘：利用 CPU 多线程并发写图，降低解码主循环阻塞.

    通过限制 in-flight 任务数避免长视频场景下 futures 无界增长占用过多内存。
    """

    def __init__(self, max_workers=1, max_pending=0):
        self.max_workers = max(1, int(max_workers))
        self.max_pending = max(0, int(max_pending))
        self.executor = None
        self.futures = deque()
        if self.max_workers > 1:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            if self.max_pending == 0:
                self.max_pending = self.max_workers * 4

    @staticmethod
    def _write_file(fpath, img, fmt, quality):
        if fmt == "jpg":
            ok = cv2.imwrite(fpath, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            ok = cv2.imwrite(fpath, img)
        return ok

    def submit(self, fpath, img, fmt, quality):
        if self.executor is None:
            ok = self._write_file(fpath, img, fmt, quality)
            if not ok:
                raise RuntimeError(f"写入失败: {fpath}")
            return
        fut = self.executor.submit(self._write_file, fpath, img.copy(), fmt, quality)
        self.futures.append((fpath, fut))
        if len(self.futures) >= self.max_pending:
            self._drain_one(wait=True)

    def _drain_one(self, wait=False):
        if not self.futures:
            return
        if wait:
            fpath, fut = self.futures.popleft()
            if not fut.result():
                raise RuntimeError(f"异步写入失败: {fpath}")
            return
        while self.futures and self.futures[0][1].done():
            fpath, fut = self.futures.popleft()
            if not fut.result():
                raise RuntimeError(f"异步写入失败: {fpath}")

    def close(self):
        while self.futures:
            self._drain_one(wait=True)
        if self.executor is not None:
            self.executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# ROS1 解码
# ---------------------------------------------------------------------------

def decode_bag_ros1(
    bag_path,
    topic,
    output_dir,
    fmt="jpg",
    quality=95,
    max_frames=None,
    h265_context_packets=24,
    frame_step=1,
    write_workers=1,
    ffmpeg_threads=0,
    ffmpeg_hwaccel="auto",
):
    """从 ROS1 bag 导出图像帧."""
    rosbag = _try_import_rosbag()
    if rosbag is None:
        raise ImportError("需要安装 rosbag: pip install rosbag")
    bag = rosbag.Bag(str(bag_path), "r")
    msg_count = bag.get_message_count(topic)

    os.makedirs(output_dir, exist_ok=True)
    index_rows = []
    frame_idx = 0
    packet_decoder = H265PacketDecoder(
        context_packets=h265_context_packets,
        ffmpeg_threads=ffmpeg_threads,
        ffmpeg_hwaccel=ffmpeg_hwaccel,
    )
    warned_ffmpeg = False
    writer = AsyncFrameWriter(max_workers=write_workers)

    frame_step = max(1, int(frame_step))
    decoded_idx = 0
    try:
        for _, msg, t in tqdm(bag.read_messages(topics=[topic]),
                              total=msg_count, desc=f"ROS1 {Path(bag_path).name}"):
            if max_frames and frame_idx >= max_frames:
                break
            img = decode_image_msg(msg)
            if img is None:
                img = maybe_decode_packet_message(msg, None, packet_decoder)
                if img is None and extract_packet_payload(msg) and not packet_decoder.ffmpeg_available and not warned_ffmpeg:
                    print("[WARN] 检测到可能的视频包消息，但系统未安装 ffmpeg，无法解码 H.265。")
                    warned_ffmpeg = True
            if img is None:
                continue
            if decoded_idx % frame_step != 0:
                decoded_idx += 1
                continue
            ts = t.to_sec() if hasattr(t, "to_sec") else float(t)
            fname = f"{frame_idx:06d}.{fmt}"
            fpath = os.path.join(output_dir, fname)
            writer.submit(fpath, img, fmt, quality)
            index_rows.append({"frame_idx": frame_idx, "timestamp": ts, "filename": fname})
            decoded_idx += 1
            frame_idx += 1
    finally:
        bag.close()
        writer.close()
    return index_rows


# ---------------------------------------------------------------------------
# ROS2 解码
# ---------------------------------------------------------------------------

def decode_bag_ros2(
    bag_path,
    topic,
    output_dir,
    fmt="jpg",
    quality=95,
    max_frames=None,
    h265_context_packets=24,
    frame_step=1,
    write_workers=1,
    ffmpeg_threads=0,
    ffmpeg_hwaccel="auto",
):
    """从 ROS2 bag 导出图像帧."""
    Ros2Reader, deserialize_cdr = _try_import_rosbags()
    if Ros2Reader is None:
        raise ImportError("需要安装 rosbags: pip install rosbags")

    os.makedirs(output_dir, exist_ok=True)
    index_rows = []
    frame_idx = 0
    packet_decoder = H265PacketDecoder(
        context_packets=h265_context_packets,
        ffmpeg_threads=ffmpeg_threads,
        ffmpeg_hwaccel=ffmpeg_hwaccel,
    )
    warned_ffmpeg = False
    writer = AsyncFrameWriter(max_workers=write_workers)

    frame_step = max(1, int(frame_step))
    decoded_idx = 0
    try:
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
                    img = maybe_decode_packet_message(msg, conn.msgtype, packet_decoder)
                    if img is None and extract_packet_payload(msg) and not packet_decoder.ffmpeg_available and not warned_ffmpeg:
                        print("[WARN] 检测到可能的视频包消息，但系统未安装 ffmpeg，无法解码 H.265。")
                        warned_ffmpeg = True
                if img is None:
                    continue
                if decoded_idx % frame_step != 0:
                    decoded_idx += 1
                    continue
                ts = timestamp / 1e9  # nanoseconds -> seconds
                fname = f"{frame_idx:06d}.{fmt}"
                fpath = os.path.join(output_dir, fname)
                writer.submit(fpath, img, fmt, quality)
                index_rows.append({"frame_idx": frame_idx, "timestamp": ts, "filename": fname})
                decoded_idx += 1
                frame_idx += 1
    finally:
        writer.close()
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
    h265_context_packets = cfg["decode"].get("h265_context_packets", 24)
    frame_step = cfg["decode"].get("frame_step", 1)
    write_workers = cfg["decode"].get("write_workers", 1)
    ffmpeg_threads = cfg["decode"].get("ffmpeg_threads", 0)
    ffmpeg_hwaccel = cfg["decode"].get("ffmpeg_hwaccel", "auto")

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
        rows = decode_bag_ros1(
            bag_path,
            front_topic,
            clip_output_dir,
            fmt,
            quality,
            max_frames,
            h265_context_packets,
            frame_step,
            write_workers,
            ffmpeg_threads,
            ffmpeg_hwaccel,
        )
    else:
        rows = decode_bag_ros2(
            bag_path,
            front_topic,
            clip_output_dir,
            fmt,
            quality,
            max_frames,
            h265_context_packets,
            frame_step,
            write_workers,
            ffmpeg_threads,
            ffmpeg_hwaccel,
        )

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
