import os, time
import psutil

class RuntimeMetrics:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.started = time.perf_counter()
        self.latencies = []
        self.cpu = []
        self.peak_memory_mb = 0.0

    def record(self, latency_ms):
        self.latencies.append(float(latency_ms))
        self.cpu.append(float(self.process.cpu_percent(interval=None)))
        self.peak_memory_mb = max(self.peak_memory_mb, self.process.memory_info().rss / 1048576)

    def summary(self, frames, detections):
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        vals = sorted(self.latencies)
        def pct(p):
            return vals[min(round((len(vals)-1)*p), len(vals)-1)] if vals else 0.0
        return {
            "frames_processed": frames,
            "elapsed_seconds": round(elapsed, 3),
            "mean_inference_ms": round(sum(vals)/len(vals), 3) if vals else 0.0,
            "median_inference_ms": round(pct(.5), 3),
            "p95_inference_ms": round(pct(.95), 3),
            "mean_processing_fps": round(frames/elapsed, 3),
            "mean_detections_per_frame": round(detections/frames, 3) if frames else 0.0,
            "mean_cpu_percent": round(sum(self.cpu)/len(self.cpu), 2) if self.cpu else 0.0,
            "peak_process_memory_mb": round(self.peak_memory_mb, 2),
        }
