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
import select
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def is_valid_frame(
    img,
    enabled=False,
    gray_ratio_max=0.92,
    saturation_mean_min=16.0,
    luma_std_min=8.0,
):
    """判断帧是否为“正常可用帧”.

    经验规则（面向“大面积灰色异常帧”）：
    1) 低饱和像素占比过高（gray_ratio）；
    2) 全图平均饱和度过低（sat_mean）；
    3) 明度标准差过低（luma_std，纹理不足）。

    返回: (is_valid, metrics_dict)
    """
    if img is None:
        return False, {
            "gray_ratio": 1.0,
            "sat_mean": 0.0,
            "luma_std": 0.0,
            "reason": "empty",
        }
    if not enabled:
        return True, {
            "gray_ratio": 0.0,
            "sat_mean": 0.0,
            "luma_std": 0.0,
            "reason": "disabled",
        }

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # 饱和度阈值可粗略刻画“灰度感”
    gray_ratio = float((sat <= 12.0).mean())
    sat_mean = float(sat.mean())
    luma_std = float(val.std())

    valid = (
        gray_ratio <= float(gray_ratio_max)
        and sat_mean >= float(saturation_mean_min)
        and luma_std >= float(luma_std_min)
    )
    reason = "ok" if valid else "gray_or_low_texture"
    return valid, {
        "gray_ratio": gray_ratio,
        "sat_mean": sat_mean,
        "luma_std": luma_std,
        "reason": reason,
    }


class H265PacketDecoder:
    """基于 ffmpeg 的 H.265 包解码器（单包 + 上下文拼接回退）."""

    def __init__(
        self,
        context_packets=24,
        ffmpeg_threads=0,
        ffmpeg_hwaccel="auto",
        decode_cooldown_packets=0,
        decoder_mode="legacy",
        legacy_timeout_sec=15.0,
        persistent_write_timeout_sec=1.0,
        persistent_read_timeout_sec=1.0,
    ):
        self.context_packets = max(1, int(context_packets))
        self.packet_buffer = deque(maxlen=self.context_packets)
        self.ffmpeg_available = shutil.which("ffmpeg") is not None
        self.ffmpeg_threads = max(0, int(ffmpeg_threads))
        self.ffmpeg_hwaccel = (ffmpeg_hwaccel or "").strip().lower()
        self.hwaccel_enabled = self.ffmpeg_hwaccel not in ("", "none", "off", "cpu")
        self.hwaccel_ready = self.ffmpeg_available and self.hwaccel_enabled
        self.decode_cooldown_packets = max(0, int(decode_cooldown_packets))
        self.decoder_mode = (decoder_mode or "legacy").strip().lower()
        if self.decoder_mode not in ("legacy", "persistent"):
            self.decoder_mode = "legacy"
        self.legacy_timeout_sec = max(0.1, float(legacy_timeout_sec))
        self.persistent_write_timeout_sec = max(0.05, float(persistent_write_timeout_sec))
        self.persistent_read_timeout_sec = max(0.05, float(persistent_read_timeout_sec))

        self.proc = None
        self.stdout_buffer = bytearray()
        self._cooldown_counter = 0
        self.stats = {
            "packet_total": 0,
            "packet_decode_attempts": 0,
            "packet_decode_skips": 0,
            "packet_decode_success": 0,
            "ffmpeg_calls": 0,
            "ffmpeg_hw_fallbacks": 0,
            "ffmpeg_process_restarts": 0,
            "ffmpeg_process_launches": 0,
            "ffmpeg_write_timeouts": 0,
            "ffmpeg_read_timeouts": 0,
        }

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
        if self.decoder_mode == "persistent":
            return self._decode_stream_persistent(stream_bytes)

        self.stats["ffmpeg_calls"] += 1

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
                timeout=self.legacy_timeout_sec,
            )
        except Exception:
            return None
        if proc.returncode != 0 or not proc.stdout:
            if self.hwaccel_ready:
                self.hwaccel_ready = False
                self.stats["ffmpeg_hw_fallbacks"] += 1
                return self._decode_stream(stream_bytes)
            return None
        arr = np.frombuffer(proc.stdout, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _extract_jpeg_from_buffer(buf):
        """从字节缓存中提取一张完整 JPEG，返回 bytes 或 None。"""
        soi = buf.find(b"\xff\xd8")
        if soi < 0:
            if len(buf) > 1024 * 1024:
                del buf[:-1024]
            return None
        eoi = buf.find(b"\xff\xd9", soi + 2)
        if eoi < 0:
            if soi > 0:
                del buf[:soi]
            return None
        jpeg = bytes(buf[soi:eoi + 2])
        del buf[:eoi + 2]
        return jpeg

    def _build_persistent_cmd(self):
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if self.ffmpeg_threads > 0:
            cmd.extend(["-threads", str(self.ffmpeg_threads)])
        if self.hwaccel_ready:
            cmd.extend(["-hwaccel", self.ffmpeg_hwaccel])
        cmd.extend([
            "-f",
            "hevc",
            "-i",
            "pipe:0",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ])
        return cmd

    def _ensure_persistent_proc(self):
        if self.proc is not None and self.proc.poll() is None:
            return True
        if not self.ffmpeg_available:
            return False
        self._close_persistent_proc()
        try:
            self.proc = subprocess.Popen(
                self._build_persistent_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self.stats["ffmpeg_process_launches"] += 1
            self.stdout_buffer.clear()
            return True
        except Exception:
            self.proc = None
            return False

    def _close_persistent_proc(self):
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=0.5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    def _decode_stream_persistent(self, stream_bytes, read_frame=True):
        if not self._ensure_persistent_proc():
            return None

        def _drain_stdout_nonblocking(max_bytes=4 * 1024 * 1024):
            """非阻塞回收 stdout，避免 feed-only 时 pipe 积压导致 ffmpeg 阻塞。"""
            if self.proc is None or self.proc.stdout is None:
                return
            fd = self.proc.stdout.fileno()
            drained = 0
            # feed-only 期间不使用历史输出，先清空缓存避免后续读到过期帧
            self.stdout_buffer.clear()
            while drained < max_bytes:
                try:
                    ready, _, _ = select.select([fd], [], [], 0)
                except Exception:
                    return
                if not ready:
                    return
                try:
                    chunk = os.read(fd, min(64 * 1024, max_bytes - drained))
                except Exception:
                    return
                if not chunk:
                    return
                drained += len(chunk)
                # feed-only 期间直接丢弃输出，避免旧帧污染与缓存增长

        def _attempt_once(payload):
            if self.proc is None or self.proc.stdin is None:
                return None, "stdin_unavailable"
            stdin_fd = self.proc.stdin.fileno()
            total = len(payload)
            sent = 0
            write_deadline = time.perf_counter() + self.persistent_write_timeout_sec
            try:
                while sent < total:
                    remain = write_deadline - time.perf_counter()
                    if remain <= 0:
                        self.stats["ffmpeg_write_timeouts"] += 1
                        return None, "write_timeout"
                    ready, _, _ = select.select([], [stdin_fd], [], min(0.05, remain))
                    if not ready:
                        continue
                    chunk_end = min(sent + 64 * 1024, total)
                    n = os.write(stdin_fd, payload[sent:chunk_end])
                    if n <= 0:
                        return None, "write_failed"
                    sent += n
            except Exception:
                return None, "write_failed"
            if not read_frame:
                _drain_stdout_nonblocking()
                return None, "feed_only"

            fd = self.proc.stdout.fileno() if self.proc and self.proc.stdout else None
            if fd is None:
                return None, "stdout_unavailable"
            timeout_sec = self.persistent_read_timeout_sec
            deadline = time.perf_counter() + timeout_sec
            while time.perf_counter() < deadline:
                try:
                    ready, _, _ = select.select([fd], [], [], 0.02)
                except Exception:
                    return None, "select_failed"
                if not ready:
                    continue
                try:
                    chunk = os.read(fd, 64 * 1024)
                except Exception:
                    return None, "read_failed"
                if not chunk:
                    return None, "empty_read"
                self.stdout_buffer.extend(chunk)
                jpeg = self._extract_jpeg_from_buffer(self.stdout_buffer)
                if jpeg is None:
                    continue
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img, None
            self.stats["ffmpeg_read_timeouts"] += 1
            return None, "timeout"

        self.stats["ffmpeg_calls"] += 1
        img, err = _attempt_once(stream_bytes)
        if err == "feed_only":
            return None
        # timeout 是“当前packet尚未形成完整可输出帧”的常见情形，不应重启并丢失解码上下文
        if err == "timeout":
            return None
        if img is not None:
            return img

        self.stats["ffmpeg_process_restarts"] += 1
        self._close_persistent_proc()

        # 尝试一次重启（如硬件加速导致异常，则降级到 CPU 再起进程）
        if self.hwaccel_ready:
            self.hwaccel_ready = False
            self.stats["ffmpeg_hw_fallbacks"] += 1
        if not self._ensure_persistent_proc():
            return None
        self.stats["ffmpeg_calls"] += 1
        img, err = _attempt_once(stream_bytes)
        if err == "feed_only":
            return None
        return img

    def decode_packet(self, packet_bytes):
        if not packet_bytes:
            return None
        self.stats["packet_total"] += 1
        self.packet_buffer.append(packet_bytes)
        # persistent 模式依赖连续输入，不可在冷却期直接丢包；冷却期只做 feed 不取帧
        if self.decoder_mode == "persistent":
            if self._cooldown_counter > 0:
                self._cooldown_counter -= 1
                self.stats["packet_decode_skips"] += 1
                self._decode_stream_persistent(packet_bytes, read_frame=False)
                return None
            self.stats["packet_decode_attempts"] += 1
            img = self._decode_stream_persistent(packet_bytes, read_frame=True)
            if img is not None:
                self.stats["packet_decode_success"] += 1
                return img
            if self.decode_cooldown_packets > 0:
                self._cooldown_counter = self.decode_cooldown_packets
            return None

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            self.stats["packet_decode_skips"] += 1
            return None

        self.stats["packet_decode_attempts"] += 1
        img = self._decode_stream(packet_bytes)
        if img is not None:
            self.stats["packet_decode_success"] += 1
            return img
        merged = b"".join(self.packet_buffer)
        img = self._decode_stream(merged)
        if img is not None:
            self.stats["packet_decode_success"] += 1
            return img
        if self.decode_cooldown_packets > 0:
            self._cooldown_counter = self.decode_cooldown_packets
        return None

    def close(self):
        self._close_persistent_proc()


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


def _safe_close_resources(bag=None, writer=None):
    """关闭资源，避免清理阶段异常覆盖主流程异常。"""
    close_errors = []
    if writer is not None:
        try:
            writer.close()
        except Exception as e:
            close_errors.append(f"writer.close: {e}")
    if bag is not None:
        try:
            bag.close()
        except Exception as e:
            close_errors.append(f"bag.close: {e}")
    return close_errors


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
    h265_decode_cooldown_packets=0,
    h265_decoder_mode="legacy",
    h265_legacy_timeout_sec=15.0,
    h265_persistent_write_timeout_sec=1.0,
    h265_persistent_read_timeout_sec=1.0,
    frame_quality_filter_enabled=False,
    frame_quality_check_interval=1,
    frame_phase_probe_enabled=False,
    frame_phase_probe_frames=20,
    gray_ratio_max=0.92,
    saturation_mean_min=16.0,
    luma_std_min=8.0,
    progress_position=0,
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
        decode_cooldown_packets=h265_decode_cooldown_packets,
        decoder_mode=h265_decoder_mode,
        legacy_timeout_sec=h265_legacy_timeout_sec,
        persistent_write_timeout_sec=h265_persistent_write_timeout_sec,
        persistent_read_timeout_sec=h265_persistent_read_timeout_sec,
    )
    warned_ffmpeg = False
    writer = AsyncFrameWriter(max_workers=write_workers)

    frame_step = max(1, int(frame_step))
    frame_quality_check_interval = max(1, int(frame_quality_check_interval))
    frame_phase_probe_frames = max(1, int(frame_phase_probe_frames))
    decoded_idx = 0
    quality_idx = 0
    skipped_bad_frames = 0
    skipped_quality_checks = 0
    phase_probe_count = 0
    phase_probe_hits = 0
    sample_phase = 0
    phase_locked = not bool(frame_phase_probe_enabled)
    main_error = None
    try:
        for _, msg, t in tqdm(
            bag.read_messages(topics=[topic]),
            total=msg_count,
            desc=f"ROS1 {Path(bag_path).name}",
            position=max(0, int(progress_position)),
            leave=True,
            dynamic_ncols=True,
        ):
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

            # 启动阶段：仅用前若干帧探测“从哪一帧开始正常”，锁定 frame_step 的相位
            if frame_phase_probe_enabled and not phase_locked:
                valid, _ = is_valid_frame(
                    img,
                    enabled=frame_quality_filter_enabled,
                    gray_ratio_max=gray_ratio_max,
                    saturation_mean_min=saturation_mean_min,
                    luma_std_min=luma_std_min,
                )
                phase_probe_count += 1
                if valid:
                    phase_probe_hits += 1
                    sample_phase = decoded_idx % frame_step
                    phase_locked = True
                if not phase_locked and phase_probe_count >= frame_phase_probe_frames:
                    phase_locked = True
                decoded_idx += 1
                continue

            do_quality_check = (quality_idx % frame_quality_check_interval == 0)
            quality_idx += 1
            if frame_quality_filter_enabled and do_quality_check and not frame_phase_probe_enabled:
                valid, _ = is_valid_frame(
                    img,
                    enabled=frame_quality_filter_enabled,
                    gray_ratio_max=gray_ratio_max,
                    saturation_mean_min=saturation_mean_min,
                    luma_std_min=luma_std_min,
                )
            else:
                valid = True
                if frame_quality_filter_enabled:
                    skipped_quality_checks += 1
            if not valid:
                skipped_bad_frames += 1
                continue
            if decoded_idx % frame_step != sample_phase:
                decoded_idx += 1
                continue
            ts = t.to_sec() if hasattr(t, "to_sec") else float(t)
            fname = f"{frame_idx:06d}.{fmt}"
            fpath = os.path.join(output_dir, fname)
            writer.submit(fpath, img, fmt, quality)
            index_rows.append({"frame_idx": frame_idx, "timestamp": ts, "filename": fname})
            decoded_idx += 1
            frame_idx += 1
    except Exception as e:
        main_error = e
        raise
    finally:
        packet_decoder.close()
        close_errors = _safe_close_resources(bag=bag, writer=writer)
        if close_errors and main_error is not None:
            print(f"[WARN] 清理阶段出现异常（已保留主异常）: {'; '.join(close_errors)}")
        elif close_errors:
            raise RuntimeError(f"清理阶段异常: {'; '.join(close_errors)}")
    packet_decoder.stats["skipped_bad_frames"] = skipped_bad_frames
    packet_decoder.stats["skipped_quality_checks"] = skipped_quality_checks
    packet_decoder.stats["phase_probe_frames"] = phase_probe_count
    packet_decoder.stats["phase_probe_hits"] = phase_probe_hits
    packet_decoder.stats["sample_phase"] = sample_phase
    return index_rows, packet_decoder.stats


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
    h265_decode_cooldown_packets=0,
    h265_decoder_mode="legacy",
    h265_legacy_timeout_sec=15.0,
    h265_persistent_write_timeout_sec=1.0,
    h265_persistent_read_timeout_sec=1.0,
    frame_quality_filter_enabled=False,
    frame_quality_check_interval=1,
    frame_phase_probe_enabled=False,
    frame_phase_probe_frames=20,
    gray_ratio_max=0.92,
    saturation_mean_min=16.0,
    luma_std_min=8.0,
    progress_position=0,
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
        decode_cooldown_packets=h265_decode_cooldown_packets,
        decoder_mode=h265_decoder_mode,
        legacy_timeout_sec=h265_legacy_timeout_sec,
        persistent_write_timeout_sec=h265_persistent_write_timeout_sec,
        persistent_read_timeout_sec=h265_persistent_read_timeout_sec,
    )
    warned_ffmpeg = False
    writer = AsyncFrameWriter(max_workers=write_workers)

    frame_step = max(1, int(frame_step))
    frame_quality_check_interval = max(1, int(frame_quality_check_interval))
    frame_phase_probe_frames = max(1, int(frame_phase_probe_frames))
    decoded_idx = 0
    quality_idx = 0
    skipped_bad_frames = 0
    skipped_quality_checks = 0
    phase_probe_count = 0
    phase_probe_hits = 0
    sample_phase = 0
    phase_locked = not bool(frame_phase_probe_enabled)
    main_error = None
    try:
        with Ros2Reader(bag_path) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            if not connections:
                raise ValueError(f"Topic {topic} not found in bag")
            conn = connections[0]
            for _, timestamp, rawdata in tqdm(
                reader.messages(connections=[conn]),
                desc=f"ROS2 {Path(bag_path).name}",
                position=max(0, int(progress_position)),
                leave=True,
                dynamic_ncols=True,
            ):
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
                # 启动阶段：仅用前若干帧探测“从哪一帧开始正常”，锁定 frame_step 的相位
                if frame_phase_probe_enabled and not phase_locked:
                    valid, _ = is_valid_frame(
                        img,
                        enabled=frame_quality_filter_enabled,
                        gray_ratio_max=gray_ratio_max,
                        saturation_mean_min=saturation_mean_min,
                        luma_std_min=luma_std_min,
                    )
                    phase_probe_count += 1
                    if valid:
                        phase_probe_hits += 1
                        sample_phase = decoded_idx % frame_step
                        phase_locked = True
                    if not phase_locked and phase_probe_count >= frame_phase_probe_frames:
                        phase_locked = True
                    decoded_idx += 1
                    continue
                do_quality_check = (quality_idx % frame_quality_check_interval == 0)
                quality_idx += 1
                if frame_quality_filter_enabled and do_quality_check and not frame_phase_probe_enabled:
                    valid, _ = is_valid_frame(
                        img,
                        enabled=frame_quality_filter_enabled,
                        gray_ratio_max=gray_ratio_max,
                        saturation_mean_min=saturation_mean_min,
                        luma_std_min=luma_std_min,
                    )
                else:
                    valid = True
                    if frame_quality_filter_enabled:
                        skipped_quality_checks += 1
                if not valid:
                    skipped_bad_frames += 1
                    continue
                if decoded_idx % frame_step != sample_phase:
                    decoded_idx += 1
                    continue
                ts = timestamp / 1e9  # nanoseconds -> seconds
                fname = f"{frame_idx:06d}.{fmt}"
                fpath = os.path.join(output_dir, fname)
                writer.submit(fpath, img, fmt, quality)
                index_rows.append({"frame_idx": frame_idx, "timestamp": ts, "filename": fname})
                decoded_idx += 1
                frame_idx += 1
    except Exception as e:
        main_error = e
        raise
    finally:
        packet_decoder.close()
        close_errors = _safe_close_resources(writer=writer)
        if close_errors and main_error is not None:
            print(f"[WARN] 清理阶段出现异常（已保留主异常）: {'; '.join(close_errors)}")
        elif close_errors:
            raise RuntimeError(f"清理阶段异常: {'; '.join(close_errors)}")
    packet_decoder.stats["skipped_bad_frames"] = skipped_bad_frames
    packet_decoder.stats["skipped_quality_checks"] = skipped_quality_checks
    packet_decoder.stats["phase_probe_frames"] = phase_probe_count
    packet_decoder.stats["phase_probe_hits"] = phase_probe_hits
    packet_decoder.stats["sample_phase"] = sample_phase
    return index_rows, packet_decoder.stats


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


def decode_single_bag(bag_path, output_dir, cfg, progress_position=0):
    """解码单个 bag，自动检测格式，返回帧索引."""
    t0 = time.perf_counter()
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
    h265_decode_cooldown_packets = cfg["decode"].get("h265_decode_cooldown_packets", 0)
    h265_decoder_mode = cfg["decode"].get("h265_decoder_mode", "legacy")
    h265_legacy_timeout_sec = cfg["decode"].get("h265_legacy_timeout_sec", 15.0)
    h265_persistent_write_timeout_sec = cfg["decode"].get("h265_persistent_write_timeout_sec", 1.0)
    h265_persistent_read_timeout_sec = cfg["decode"].get("h265_persistent_read_timeout_sec", 1.0)
    frame_quality_filter_enabled = bool(cfg["decode"].get("frame_quality_filter_enabled", False))
    frame_quality_check_interval = int(cfg["decode"].get("frame_quality_check_interval", 1))
    frame_phase_probe_enabled = bool(cfg["decode"].get("frame_phase_probe_enabled", True))
    frame_phase_probe_frames = int(cfg["decode"].get("frame_phase_probe_frames", 20))
    gray_ratio_max = float(cfg["decode"].get("gray_ratio_max", 0.92))
    saturation_mean_min = float(cfg["decode"].get("saturation_mean_min", 16.0))
    luma_std_min = float(cfg["decode"].get("luma_std_min", 8.0))

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
        elapsed_sec = time.perf_counter() - t0
        return {
            "bag": bag_path.name,
            "bag_path": str(bag_path),
            "frames": 0,
            "status": "failed",
            "elapsed_sec": elapsed_sec,
            "fps_out": 0.0,
            "packet_stats": {},
        }

    front_topic = find_front_topic(topics, candidate_topics)
    if front_topic is None:
        print(f"[ERROR] 未找到前视 topic, 可用 topics: {topics}")
        elapsed_sec = time.perf_counter() - t0
        return {
            "bag": bag_path.name,
            "bag_path": str(bag_path),
            "frames": 0,
            "status": "failed",
            "elapsed_sec": elapsed_sec,
            "fps_out": 0.0,
            "packet_stats": {},
        }

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
            h265_decode_cooldown_packets,
            h265_decoder_mode,
            h265_legacy_timeout_sec,
            h265_persistent_write_timeout_sec,
            h265_persistent_read_timeout_sec,
            frame_quality_filter_enabled,
            frame_quality_check_interval,
            frame_phase_probe_enabled,
            frame_phase_probe_frames,
            gray_ratio_max,
            saturation_mean_min,
            luma_std_min,
            progress_position,
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
            h265_decode_cooldown_packets,
            h265_decoder_mode,
            h265_legacy_timeout_sec,
            h265_persistent_write_timeout_sec,
            h265_persistent_read_timeout_sec,
            frame_quality_filter_enabled,
            frame_quality_check_interval,
            frame_phase_probe_enabled,
            frame_phase_probe_frames,
            gray_ratio_max,
            saturation_mean_min,
            luma_std_min,
            progress_position,
        )
    rows, packet_stats = rows

    # 保存时间戳索引
    if rows:
        index_path = os.path.join(clip_output_dir, "timestamp_index.csv")
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_idx", "timestamp", "filename"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  导出 {len(rows)} 帧 -> {clip_output_dir}")

    elapsed_sec = time.perf_counter() - t0
    return {
        "bag": bag_path.name,
        "bag_path": str(bag_path),
        "frames": len(rows),
        "status": "ok" if rows else "failed",
        "elapsed_sec": elapsed_sec,
        "fps_out": (len(rows) / elapsed_sec) if elapsed_sec > 0 else 0.0,
        "packet_stats": packet_stats,
    }


def _decode_single_bag_worker(bag_path, output_dir, cfg, progress_position):
    return decode_single_bag(bag_path, output_dir, cfg, progress_position=progress_position)


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

    bag_workers = int(cfg["decode"].get("bag_workers", 1))
    bag_workers = max(1, bag_workers)
    print(f"找到 {len(bag_files)} 个 bag 文件")
    print(f"bag_workers={bag_workers}（bag 级并行进程数）")
    summary = []
    if bag_workers == 1 or len(bag_files) == 1:
        for bag_path in bag_files:
            print(f"\n处理: {bag_path}")
            report = decode_single_bag(bag_path, output_dir, cfg, progress_position=0)
            summary.append(report)
    else:
        worker_count = min(bag_workers, len(bag_files))
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            fut_to_bag = {
                ex.submit(
                    _decode_single_bag_worker,
                    str(bag_path),
                    output_dir,
                    cfg,
                    idx % worker_count,
                ): bag_path
                for idx, bag_path in enumerate(bag_files)
            }
            for fut in as_completed(fut_to_bag):
                bag_path = fut_to_bag[fut]
                try:
                    report = fut.result()
                    summary.append(report)
                    print(
                        f"\n完成: {bag_path.name} | frames={report['frames']} | "
                        f"time={report['elapsed_sec']:.2f}s | fps={report['fps_out']:.2f}"
                    )
                except Exception as e:
                    print(f"\n[ERROR] 处理失败: {bag_path} -> {e}")
                    summary.append({
                        "bag": bag_path.name,
                        "bag_path": str(bag_path),
                        "frames": 0,
                        "status": "failed",
                        "elapsed_sec": 0.0,
                        "fps_out": 0.0,
                        "packet_stats": {},
                    })

    summary = sorted(summary, key=lambda x: x["bag"])

    # 输出汇总
    print("\n===== 解码汇总 =====")
    for s in summary:
        print(
            f"  {s['bag']}: {s['frames']} 帧 [{s['status']}] | "
            f"time={s['elapsed_sec']:.2f}s | fps={s['fps_out']:.2f}"
        )
        p = s.get("packet_stats") or {}
        if p:
            print(
                "    packet_total={packet_total}, packet_decode_attempts={packet_decode_attempts}, "
                "packet_decode_skips={packet_decode_skips}, packet_decode_success={packet_decode_success}, "
                "ffmpeg_calls={ffmpeg_calls}, ffmpeg_hw_fallbacks={ffmpeg_hw_fallbacks}, "
                "ffmpeg_process_launches={ffmpeg_process_launches}, "
                "ffmpeg_process_restarts={ffmpeg_process_restarts}, "
                "skipped_bad_frames={skipped_bad_frames}, "
                "skipped_quality_checks={skipped_quality_checks}, "
                "phase_probe_frames={phase_probe_frames}, "
                "phase_probe_hits={phase_probe_hits}, "
                "sample_phase={sample_phase}".format(**p)
            )

    ok_count = sum(1 for s in summary if s["status"] == "ok")
    print(f"\n成功: {ok_count}/{len(summary)}")


if __name__ == "__main__":
    main()
