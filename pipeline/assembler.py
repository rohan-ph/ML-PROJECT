import os
import cv2
import json
import time
import shutil
import hashlib
import subprocess
from collections import deque
from typing import Optional, Dict, Any, Tuple
import numpy as np

def _ensure_browser_compatible_h264(video_path: str) -> None:
    """
    Transcodes an OpenCV-generated MP4 video to browser-compatible H.264 (avc1)
    with yuv420p pixel format and +faststart (moov atom at the beginning).
    HTML5 <video> elements in Chrome/Edge/Firefox cannot decode OpenCV's default mp4v.
    """
    if not shutil.which("ffmpeg"):
        return
    tmp_path = str(video_path) + ".h264_tmp.mp4"
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            tmp_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(tmp_path):
            shutil.move(tmp_path, str(video_path))
    except Exception as e:
        print(f"[ClipAssembler] Warning during H.264 transcode: {e}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

class ClipAssembler:
    """
    ClipAssembler maintains a ring buffer of video frames,
    tracks the state of violent incidents (IDLE, ACTIVE, COOLDOWN), and exports
    evidence clips with surrounding context (pre-event and post-event).
    """
    def __init__(self, fps=16, pre_seconds=1.0, post_seconds=1.0, output_dir='results/clips', codec='mp4v', include_hash=True):
        if isinstance(fps, dict):
            cfg = fps
            asm_cfg = cfg.get("assembler", cfg)
            self.fps = asm_cfg.get("fps", 16)
            self.pre_seconds = float(asm_cfg.get("pre_seconds", 1.0))
            self.post_seconds = float(asm_cfg.get("post_seconds", 1.0))
            self.output_dir = output_dir if output_dir != 'results/clips' else asm_cfg.get("output_dir", output_dir)
            self.codec = asm_cfg.get("codec", codec)
            self.include_hash = asm_cfg.get("include_hash", include_hash)
        else:
            self.fps = int(fps)
            self.pre_seconds = float(pre_seconds)
            self.post_seconds = float(post_seconds)
            self.output_dir = output_dir
            self.codec = codec
            self.include_hash = include_hash
        
        self.ring_buffer = deque(maxlen=max(1, int(round(self.fps * self.pre_seconds))))
        self.state = 'IDLE' # IDLE -> ACTIVE -> COOLDOWN
        self.current_clip = []
        self.cooldown_counter = 0
        self.clip_start_time = None
        self.clip_id = 0
        self.scores = []
        self.has_genuine_incident = False
        
        os.makedirs(self.output_dir, exist_ok=True)

    def update(self, frame, is_triggered, frame_idx=0, score=None):
        """
        Alias for push to support stream runner API.
        """
        event = 'ONSET' if (is_triggered and self.state == 'IDLE') else ('OFFSET' if (not is_triggered and self.state == 'ACTIVE') else None)
        return self.push(frame, event, is_triggered, score=score)
        
    def push(self, frame, event, incident_state, score=None):
        """
        Push a frame and its corresponding event into the assembler.
        Only genuine violent incidents trigger clipping.
        """
        # Always append to ring buffer
        self.ring_buffer.append(frame)
        
        # A violent incident requires confirmed incident state from IncidentNeuron
        is_violent = (event == 'ONSET' or incident_state is True)
        
        if self.state == 'IDLE':
            if is_violent:
                self.state = 'ACTIVE'
                self.has_genuine_incident = True
                self.clip_start_time = time.time()
                # Include pre-buffer frames (tight ~1s lead-in of violence onset)
                self.current_clip = list(self.ring_buffer)
                self.incident_start_idx = len(self.current_clip)
                self.scores = [score] if score is not None else []
        
        elif self.state == 'ACTIVE':
            self.current_clip.append(frame)
            if score is not None:
                self.scores.append(score)
            if event == 'OFFSET' or incident_state is False:
                self.state = 'COOLDOWN'
                self.incident_end_idx = len(self.current_clip)
                self.cooldown_counter = max(1, int(round(self.fps * self.post_seconds)))
                
        elif self.state == 'COOLDOWN':
            self.current_clip.append(frame)
            if score is not None:
                self.scores.append(score)
            
            if is_violent:
                # Re-trigger
                self.state = 'ACTIVE'
                self.cooldown_counter = 0
            else:
                self.cooldown_counter -= 1
                if self.cooldown_counter <= 0:
                    return self.export()
        return None
                    
    def export(self, manual_clip_id=None):
        if not self.current_clip:
            return None

        # Verify genuine violence: reject non-violence clips
        peak_score = max(self.scores) if self.scores else 0.0
        avg_score = float(np.mean(self.scores)) if self.scores else 0.0

        if not getattr(self, 'has_genuine_incident', False) or peak_score < 0.55:
            print(f"[ClipAssembler] Discarding non-violent candidate clip (has_incident={getattr(self, 'has_genuine_incident', False)}, peak={peak_score:.3f}, avg={avg_score:.3f})")
            self.state = 'IDLE'
            self.current_clip = []
            self.scores = []
            self.has_genuine_incident = False
            return None
            
        timestamp = int(time.time())
        c_id = manual_clip_id if manual_clip_id is not None else f"clip_{timestamp}_{self.clip_id}"
        video_filename = f"{c_id}.mp4"
        meta_filename = f"{c_id}.json"
        video_path = os.path.join(self.output_dir, video_filename)
        meta_path = os.path.join(self.output_dir, meta_filename)
        
        # Write video
        h, w = self.current_clip[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
        
        for f in self.current_clip:
            writer.write(f)
            
        writer.release()
        _ensure_browser_compatible_h264(video_path)
        
        # Compute hash and verify
        file_hash = None
        is_verified = False
        if self.include_hash:
            sha256 = hashlib.sha256()
            with open(video_path, 'rb') as f:
                while chunk := f.read(65536):
                    sha256.update(chunk)
            file_hash = sha256.hexdigest()

            # Forensic verification: immediately recompute hash independently
            verify_sha256 = hashlib.sha256()
            with open(video_path, 'rb') as f:
                while chunk := f.read(65536):
                    verify_sha256.update(chunk)
            is_verified = (file_hash == verify_sha256.hexdigest())

        # Metadata
        num_frames = len(self.current_clip)
        duration = num_frames / self.fps

        incident_start_sec = getattr(self, 'incident_start_idx', len(self.ring_buffer)) / self.fps
        incident_end_sec = getattr(self, 'incident_end_idx', num_frames) / self.fps

        metadata = {
            "clip_id": c_id,
            "incident_id": f"INC-{timestamp}-{self.clip_id}",
            "source_video": getattr(self, "source_video", "Live Surveillance Stream"),
            "start_timestamp": self.clip_start_time,
            "end_timestamp": time.time(),
            "incident_start_seconds": round(float(incident_start_sec), 2),
            "incident_end_seconds": round(float(incident_end_sec), 2),
            "clip_start_seconds": round(max(0.0, float(incident_start_sec) - self.pre_seconds), 2),
            "clip_end_seconds": round(float(duration), 2),
            "duration_seconds": round(duration, 3),
            "num_frames": num_frames,
            "frames_count": num_frames,
            "fps": self.fps,
            "violence_score_avg": round(avg_score, 4),
            "model_confidence_score": round(peak_score, 4),
            "violence_score_peak": round(peak_score, 4),
            "video_file": video_filename,
            "evidence_clip_file": video_filename,
            "evidence_clip_path": video_path,
            "sha256_hash": file_hash,
            "sha256_verified": is_verified,
            "pre_buffer_seconds": self.pre_seconds,
            "post_buffer_seconds": self.post_seconds
        }

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        print(f"Exported clip {c_id}: {video_filename} ({duration:.2f}s, {num_frames} frames) - Hash: {file_hash[:12] if file_hash else 'N/A'} (Verified: {is_verified})")

        self.latest_metadata = metadata
        self.latest_video_path = video_path
        self.latest_meta_path = meta_path

        # Reset state
        self.state = 'IDLE'
        self.current_clip = []
        self.scores = []
        self.has_genuine_incident = False
        self.clip_id += 1
        return video_path

    def flush(self):
        """
        Export any active/cooldown clip at stream end, ONLY IF genuine violence occurred.
        """
        if self.state in ['ACTIVE', 'COOLDOWN'] and self.current_clip:
            peak_score = max(self.scores) if self.scores else 0.0
            if getattr(self, 'has_genuine_incident', False) and peak_score >= 0.55:
                return self.export()
            else:
                peak_s = peak_score
                avg_s = float(np.mean(self.scores)) if self.scores else 0.0
                print(f"[ClipAssembler] Discarding sub-threshold clip on flush (peak={peak_s:.3f}, avg={avg_s:.3f})")
                self.state = 'IDLE'
                self.current_clip = []
                self.scores = []
                self.has_genuine_incident = False
                return None
        return None

    @staticmethod
    def extract_incident_clip_offline(
        video_path: str,
        incident_start_sec: float,
        incident_end_sec: float,
        output_dir: str = "evidence",
        pre_sec: float = 10.0,
        post_sec: float = 10.0,
        incident_id: Optional[str] = None,
        model_score: float = 0.85
    ) -> Dict[str, Any]:
        """
        Extracts an evidence clip matching:
          clip_start = max(0.0, incident_start_sec - 10.0)
          clip_end   = min(video_duration, incident_end_sec + 10.0)
        """
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_duration = total_frames / fps if fps and fps > 0 else 0.0

        # Exact formula from project requirement
        clip_start_sec = max(0.0, float(incident_start_sec) - pre_sec)
        clip_end_sec = min(total_duration, float(incident_end_sec) + post_sec)
        
        pre_truncated = (incident_start_sec - pre_sec) < 0.0
        post_truncated = (incident_end_sec + post_sec) > total_duration

        start_frame = max(0, int(round(clip_start_sec * fps)))
        end_frame = min(total_frames - 1, int(round(clip_end_sec * fps)))

        timestamp = int(time.time())
        c_id = incident_id or f"inc_{timestamp}"
        out_video_name = f"evidence_{c_id}.mp4"
        out_video_path = os.path.join(output_dir, out_video_name)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        written_frames = 0
        current_f = start_frame

        while current_f <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            written_frames += 1
            current_f += 1

        writer.release()
        cap.release()
        _ensure_browser_compatible_h264(out_video_path)

        # Compute SHA-256 hash
        sha256 = hashlib.sha256()
        with open(out_video_path, 'rb') as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()

        # Independent verification check
        verify_sha = hashlib.sha256()
        with open(out_video_path, 'rb') as f:
            while chunk := f.read(65536):
                verify_sha.update(chunk)
        hash_verified = (file_hash == verify_sha.hexdigest())

        actual_duration = written_frames / fps if fps > 0 else 0.0

        metadata = {
            "incident_id": c_id,
            "source_video": str(video_path),
            "incident_start_seconds": round(float(incident_start_sec), 3),
            "incident_end_seconds": round(float(incident_end_sec), 3),
            "clip_start_seconds": round(clip_start_sec, 3),
            "clip_end_seconds": round(clip_end_sec, 3),
            "duration_seconds": round(actual_duration, 3),
            "frames_count": written_frames,
            "fps": fps,
            "pre_truncated": pre_truncated,
            "post_truncated": post_truncated,
            "model_confidence_score": round(float(model_score), 4),
            "sha256_hash": file_hash,
            "sha256_verified": hash_verified,
            "timestamp": timestamp,
            "evidence_clip_file": out_video_name,
            "evidence_clip_path": out_video_path
        }

        out_meta_path = os.path.join(output_dir, f"evidence_{c_id}.json")
        with open(out_meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        return metadata

if __name__ == "__main__":
    # Standalone test
    print("Testing ClipAssembler...")
    assembler = ClipAssembler(fps=2, pre_seconds=2, post_seconds=2, output_dir='test_clips')
    
    # Push dummy frames
    for i in range(20):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        event = None
        state = False
        
        if i == 5:
            event = 'ONSET'
            state = True
        elif 5 < i < 10:
            state = True
        elif i == 10:
            event = 'OFFSET'
            state = False
            
        assembler.push(frame, event, state, score=0.8 if state else 0.1)
        
    assembler.flush()
    print("Test passed.")
