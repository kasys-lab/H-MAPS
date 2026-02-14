#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
demo_main.py
- H-MAPS Main Loop
- Feature: Dynamic Reading Speed Estimation
- Update: Added --demo flag for mock playback from JSONL logs (Skipping heavy init)
"""

import sys
import argparse
import signal
import hashlib
import time
import threading
import json
from collections import deque
from typing import Set, Optional, Tuple, List
import numpy as np
from pynput import mouse, keyboard

from PySide6.QtCore import QObject, Signal, Slot, QTimer, QThread
from PySide6.QtWidgets import QApplication

import demo_backend
import demo_gui

start_time = time.time()


# ================= Config =================

ADAPTIVE_WINDOW_SIZE = 10     
MIN_SIMILARITY_BOUND = 0.2    
Z_SCORE_THRESHOLD = 3.0       

MIN_DWELL_SEC = 10.0         
MAX_DWELL_SEC = 60.0         
DEFAULT_READING_SPEED_CPS = 20.0  
SPEED_HISTORY_SIZE = 20           
MIN_CHARS_FOR_SPEED_EST = 100     
MIN_DURATION_FOR_SPEED_EST = 3.0  

TRIGGER_B_REGRESSION_DWELL_SEC = 5 
HISTORY_WINDOW_SEC = 180           
REGRESSION_SIMILARITY_THRESHOLD = 0.8
MIN_SCENE_DURATION_FOR_HISTORY = 2.0

AFK_TIMEOUT_SEC = 60.0             

LOCAL_CONTEXT_CHUNK_SIZE = 2000    
SESSION_UPDATE_INTERVAL_SEC = 300  

# ================= Utils =================

def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    if not text1 and not text2: return 1.0
    if not text1 or not text2: return 0.0
    def get_bigrams(t: str) -> Set[str]:
        t = "".join(t.split())
        return set(t[i:i+2] for i in range(max(0, len(t)-1)))
    set1 = get_bigrams(text1)
    set2 = get_bigrams(text2)
    if not set1 and not set2: return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0: return 0.0
    return intersection / union

# ================= Activity Monitor =================

class ActivityMonitor:
    def __init__(self):
        self.last_activity_time = time.time()
        self._lock = threading.Lock()
        self.mouse_listener = mouse.Listener(on_move=self._on_activity, on_click=self._on_activity, on_scroll=self._on_activity)
        self.mouse_listener.start()
        self.key_listener = keyboard.Listener(on_press=self._on_activity)
        self.key_listener.start()
        print("[INFO] AFK Detection enabled.", flush=True)

    def _on_activity(self, *args):
        with self._lock:
            self.last_activity_time = time.time()

    def get_idle_time(self) -> float:
        with self._lock:
            return time.time() - self.last_activity_time

# ================= Worker =================

class PipelineWorker(QObject):
    resultsReady = Signal(list)
    finished = Signal() 

    def __init__(self, ocr: demo_backend.OCRWrapper, retriever: Optional[demo_backend.S2ORCRetriever], args, memory_manager):
        super().__init__()
        self.ocr = ocr
        self.retriever = retriever
        self.args = args
        self.memory = memory_manager
        
        self.local_context_cache = [] 
        self.last_text = ""         
        self.current_scene_hash = "" 
        self.dwell_start_time = 0.0 
        self.scene_history = deque() 
        self.accumulated_text_buffer = ""
        self.last_session_update = time.time()
        self.trigger_fired_for_current_scene = False
        self.similarity_scores = deque(maxlen=ADAPTIVE_WINDOW_SIZE)
        
        self.speed_history = deque(maxlen=SPEED_HISTORY_SIZE)
        self.current_estimated_speed = DEFAULT_READING_SPEED_CPS
        self.current_required_dwell = MIN_DWELL_SEC

        # --- DEMO DATA LOADING ---
        self.demo_payload = None
        if self.args.demo:
            try:
                print(f"[DEMO] Loading mock data from: {self.args.demo}", flush=True)
                with open(self.args.demo, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    last_line = lines[-1] if lines else "{}"
                    data = json.loads(last_line)
                    if "events" in data:
                        self.demo_payload = data["events"]
                        print(f"[DEMO] Successfully loaded {len(self.demo_payload)} events to replay.", flush=True)
                    else:
                        print("[DEMO] Warning: JSONL format seems incorrect (key 'events' missing).", flush=True)
            except Exception as e:
                print(f"[DEMO] Error loading demo file: {e}", flush=True)

    def _extract_new_lines(self, old_text: str, new_text: str) -> str:
        if not old_text: return new_text
        old_lines = {line.strip() for line in old_text.splitlines() if line.strip()}
        return "\n".join([l for l in new_text.splitlines() if l.strip() and l.strip() not in old_lines])

    def _update_reading_speed(self, text: str, duration: float):
        char_count = len(text.replace(" ", "").replace("\n", ""))
        if char_count < MIN_CHARS_FOR_SPEED_EST or duration < MIN_DURATION_FOR_SPEED_EST:
            return
        raw_speed = char_count / duration
        if raw_speed < 1.0 or raw_speed > 200.0: 
            return

        self.speed_history.append(raw_speed)
        if len(self.speed_history) >= 3:
            speeds = np.array(self.speed_history)
            mean = np.mean(speeds)
            std = np.std(speeds)
            filtered = [s for s in speeds if abs(s - mean) <= 2 * std]
            self.current_estimated_speed = np.mean(filtered) if filtered else mean
        else:
            self.current_estimated_speed = np.mean(self.speed_history)

        print(f"[SPEED] Updated Reading Speed: {self.current_estimated_speed:.1f} cps", flush=True)

    def _calculate_required_dwell(self, text: str) -> float:
        char_count = len(text.replace(" ", "").replace("\n", ""))
        estimated_time = char_count / self.current_estimated_speed
        return min(max(estimated_time, MIN_DWELL_SEC), MAX_DWELL_SEC)

    @Slot(tuple, float)
    def process_frame(self, mask_rect: Tuple[int, int, int, int], idle_sec: float):
        try:
            current_time = time.time()
            is_afk = (idle_sec > AFK_TIMEOUT_SEC)
            img = demo_backend.capture_screen_image(mask_rect=mask_rect)
            ocr_text = self.ocr.image_to_text(img).strip()
            
            if len(ocr_text) < 30: return

            sim = calculate_jaccard_similarity(self.last_text, ocr_text)
            
            is_continuous = True
            if len(self.similarity_scores) < ADAPTIVE_WINDOW_SIZE:
                is_continuous = (sim >= 0.5)
            else:
                scores = np.array(self.similarity_scores)
                dynamic_threshold = np.mean(scores) - (Z_SCORE_THRESHOLD * np.std(scores))
                final_threshold = max(dynamic_threshold, MIN_SIMILARITY_BOUND)
                is_continuous = (sim >= final_threshold)

            if is_continuous: self.similarity_scores.append(sim)
            else: self.similarity_scores.clear()

            current_snapshot_hash = hashlib.sha1(ocr_text.encode("utf-8")).hexdigest()
            while self.scene_history and (current_time - self.scene_history[0][1] > HISTORY_WINDOW_SEC):
                self.scene_history.popleft()

            if not is_continuous:
                # Scene Changed
                duration = current_time - self.dwell_start_time
                if self.last_text and not is_afk: 
                    self._update_reading_speed(self.last_text, duration)

                if self.last_text and duration > MIN_SCENE_DURATION_FOR_HISTORY:
                    self.scene_history.append((self.last_text, current_time))
                
                self.last_text = ocr_text
                self.current_scene_hash = current_snapshot_hash
                self.dwell_start_time = current_time
                self.trigger_fired_for_current_scene = False
                self.accumulated_text_buffer = ocr_text
                
                self.current_required_dwell = self._calculate_required_dwell(ocr_text)
                print(f"[H-MAPS] Scene Changed. Req Dwell: {self.current_required_dwell:.1f}s (Speed: {self.current_estimated_speed:.1f} cps)", flush=True)

            else:
                # Continuous Dwell
                dwell_duration = current_time - self.dwell_start_time
                diff = self._extract_new_lines(self.last_text, ocr_text)
                if diff: self.accumulated_text_buffer += ("\n" + diff)
                self.last_text = ocr_text

                if len(self.accumulated_text_buffer) > LOCAL_CONTEXT_CHUNK_SIZE:
                    print("[H-MAPS] Generating Local Context...", flush=True)
                    summary = demo_backend.llm_summarize_chunk(self.accumulated_text_buffer)
                    # retrieverがNoneの場合はキャッシュ生成をスキップ
                    if self.retriever:
                        try:
                            vec = self.retriever.encode_query(summary) 
                            self.local_context_cache.append({"summary": summary, "vector": vec, "timestamp": current_time})
                        except: pass
                    self.accumulated_text_buffer = ""

                if not self.trigger_fired_for_current_scene and not is_afk:
                    is_regression = any(calculate_jaccard_similarity(pt, ocr_text) >= REGRESSION_SIMILARITY_THRESHOLD for pt, _ in self.scene_history)
                    
                    if is_regression and dwell_duration >= TRIGGER_B_REGRESSION_DWELL_SEC:
                        print(f"[TRIGGER] Type B (Content Revisit) fired!", flush=True)
                        self._fire_query(ocr_text, mode="support")
                        self.trigger_fired_for_current_scene = True
                    elif dwell_duration >= self.current_required_dwell:
                        print(f"[TRIGGER] Type A (Sustained Attention) fired!", flush=True)
                        self._fire_query(ocr_text, mode="expansion")
                        self.trigger_fired_for_current_scene = True
                
            if current_time - self.last_session_update > SESSION_UPDATE_INTERVAL_SEC:
                print("[H-MAPS] Periodic Memory Update...", flush=True)
                self.last_session_update = current_time

        except Exception as e:
            print(f"[ERROR] Worker exception: {e}", flush=True)
        finally:
            self.finished.emit()

    def _fire_query(self, ocr_text: str, mode: str):
        """
        Trigger fired. 
        If --demo is ON, bypass real processing and simulate loading from log.
        """
        if self.args.demo and self.demo_payload:
            self._fire_mock_sequence(mode)
            return

        # === Real Processing Logic ===
        # Safety check: if demo mode is on but payload missing, we can't run if retriever is skipped
        if not self.retriever:
            print("[ERR] Demo mode enabled but payload is missing/empty, and Retriever is None. Cannot search.", flush=True)
            return

        t_start_total = time.time()
        print(f"[{mode.upper()}] Pipeline triggered. Starting measurement...", flush=True)

        # --- Phase 1: Context Preparation ---
        t_p1_start = time.time()
        print("[PRIVACY] Sanitizing current screen text via Local LLM...", flush=True)
        safe_current_context = demo_backend.llm_summarize_chunk(ocr_text)

        relevant_contexts = []
        try:
            current_vec = self.retriever.encode_query(ocr_text) 
            if self.local_context_cache:
                cached_vectors = np.vstack([item["vector"] for item in self.local_context_cache])
                scores = np.dot(cached_vectors, current_vec.T).flatten()
                top_k = min(3, len(scores))
                indices = np.argsort(scores)[::-1][:top_k]
                for idx in indices:
                    relevant_contexts.append(self.local_context_cache[idx]["summary"])
        except: pass

        session_ctx = self.memory.get_latest_session_context()
        core_prof = self.memory.get_inferred_profile()
        t_p1_end = time.time()


        # --- Phase 2: Question Generation (LLM) ---    
        t_p2_start = time.time()
        print(f"[DEBUG] Generating queries (Mode={mode.upper()})...", flush=True)
        questions, used_prompt = demo_backend.generate_query_with_hmaps(
            current_text=safe_current_context,
            local_context=relevant_contexts,
            session_context=session_ctx,
            inferred_profile=core_prof,
            trigger_type=mode,
            provider=self.args.gen_provider
        )
        if used_prompt: self.memory.save_prompt_log(used_prompt, trigger_type=mode)
        t_p2_end = time.time()
        
        # --- Phase 3: Retrieval (Vector Search) ---
        t_p3_start = time.time()
        payload = []
        for q in questions:
            results = self.retriever.search(q, top_k=self.args.top_k)
            cards = []
            for rec in results:
                summary = f"{rec.get('meta','')}\n{rec.get('abstract','')}"
                cards.append({
                    "paper_id": rec["paper_id"],
                    "title": rec["title"],
                    "summary": summary,
                    "url": rec.get("url"),
                })
            if cards: payload.append({"question": q, "cards": cards, "trigger_type": mode})
        t_p3_end = time.time()
        
        # --- Metrics ---
        t_total_end = time.time()
        dur_prep = t_p1_end - t_p1_start
        dur_gen = t_p2_end - t_p2_start
        dur_ret = t_p3_end - t_p3_start
        dur_total = t_total_end - t_start_total
        dur_overhead = dur_total - (dur_prep + dur_gen + dur_ret)

        print("-" * 60, flush=True)
        print(f"[PAPER-METRICS] End-to-End Latency: {dur_total:.4f} sec", flush=True)
        print(f"  1. Question Generation (LLM): {dur_gen:.4f} sec", flush=True)
        print(f"  2. Vector Encoding & Retrieval: {dur_ret:.4f} sec", flush=True)
        print(f"  3. Context Prep & Overhead:   {dur_prep + dur_overhead:.4f} sec", flush=True)
        print("-" * 60, flush=True)

        if payload:
            self.memory.save_search_log(payload)
            self.resultsReady.emit(payload)

    def _fire_mock_sequence(self, mode: str):
        """
        Simulate the processing flow using cached JSONL data.
        """
        print(f"[{mode.upper()}] Pipeline triggered (DEMO MODE).", flush=True)
        
        # 1. Simulate Context Prep
        print("[PRIVACY] Sanitizing current screen text via Local LLM...", flush=True)
        time.sleep(0.5) # Fake work
        
        # 2. Simulate LLM Generation
        print(f"[DEBUG] Generating queries (Mode={mode.upper()})...", flush=True)
        # 1.5秒待機することで「考えている」演出を入れます
        time.sleep(1.5) 
        
        # 3. Simulate Retrieval
        print("[INFO] S2ORC Search started...", flush=True)
        time.sleep(0.3) 
        
        # Construct Payload from JSONL
        gui_payload = []
        for event in self.demo_payload:
            q_text = event.get("question", "")
            raw_papers = event.get("papers", [])
            
            cards = []
            for p in raw_papers:
                cards.append({
                    "paper_id": p.get("paper_id"),
                    "title": p.get("title"),
                    "summary": p.get("summary"),
                    "url": p.get("url")
                })
            
            if cards:
                gui_payload.append({
                    "question": q_text,
                    "cards": cards,
                    "trigger_type": mode # Use the current detected trigger type
                })

        print("-" * 60, flush=True)
        print(f"[DEMO-METRICS] Simulated Latency: 2.3000 sec", flush=True)
        print("-" * 60, flush=True)

        self.resultsReady.emit(gui_payload)


# ================= Main Controller =================

class MainController(QObject):
    request_process = Signal(tuple, float) 

    def __init__(self, args, overlay):
        super().__init__()
        self.args = args
        self.overlay = overlay
        
        self.ocr = demo_backend.OCRWrapper(lang=args.ocr_lang)
        self.memory_manager = demo_backend.MemoryManager()
        if args.persona:
            self.memory_manager.inject_demo_profile(args.persona)

        # --- SKIP HEAVY LOADING IN DEMO MODE ---
        if args.demo:
            print("[INFO] DEMO MODE DETECTED. Skipping Index/LLM loading for instant startup.", flush=True)
            self.retriever = None
            # Local LLM init is also skipped
        else:
            if args.local_model:
                demo_backend.init_local_llm(args.local_model)
            
            print("[INFO] Initializing Backend objects...", flush=True)
            self.retriever = demo_backend.S2ORCRetriever(
                model_dir=args.model_dir, index_path=args.index_path, ids_path=args.ids_path, db_path=args.db_path,
                max_length=args.max_length, top_k=args.top_k,
            )

        self.activity_monitor = ActivityMonitor()
        self.thread = QThread()
        self.worker = PipelineWorker(self.ocr, self.retriever, args, self.memory_manager)
        self.worker.moveToThread(self.thread)
        self.request_process.connect(self.worker.process_frame) 
        self.worker.resultsReady.connect(overlay.update_results) 
        self.worker.finished.connect(self._on_worker_finished) 
        self.thread.start()

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._on_timer)
        self.is_worker_busy = False
        self.state = {"hover": False, "manual_pause": False}
        self.overlay.hoverChanged.connect(self._on_hover_changed)
        self.overlay.manualPauseToggled.connect(self._on_manual_pause_toggled)

    def start(self): self.timer.start()

    def _on_timer(self):
        if self.is_worker_busy or self.state["hover"] or self.state["manual_pause"]: return
        mask = self.overlay.get_mask_rect_physical()
        idle_sec = self.activity_monitor.get_idle_time()
        self.is_worker_busy = True 
        self.request_process.emit(mask, idle_sec)

    @Slot()
    def _on_worker_finished(self): self.is_worker_busy = False 

    @Slot(bool)
    def _on_hover_changed(self, hovered):
        self.state["hover"] = hovered
        self._update_timer_state()

    @Slot(bool)
    def _on_manual_pause_toggled(self, paused):
        self.state["manual_pause"] = paused
        self._update_timer_state()

    def _update_timer_state(self):
        if self.state["hover"] or self.state["manual_pause"]:
            print(f"[DEBUG] Monitoring PAUSED", flush=True)
        else:
            print("[DEBUG] Monitoring RESUMED", flush=True)

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="models/biencoder_litsearch_e5_small_v2_all_tevatron/tevatron_out")
    ap.add_argument("--index_path", type=str, default="index/S2ORC/litsearch_e5_small_v2_all_l512")
    ap.add_argument("--ids_path", type=str, default="index/S2ORC/litsearch_e5_small_v2_all_l512")
    ap.add_argument("--db_path", type=str, default="S2ORC/papers.db")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--ocr_lang", type=str, default="eng")
    ap.add_argument("--num_questions", type=int, default=3)
    ap.add_argument("--local_model", type=str, default="models/Phi-3.5-mini-instruct-Q4_K_M.gguf")
    ap.add_argument("--persona", type=str, choices=["nlp", "hci"], default=None)
    ap.add_argument("--gen_provider", type=str, choices=["openai", "local"], default="openai",
                    help="Choose 'openai' for GPT-4o or 'local' for local LLM question generation.")
    
    # Added argument for Demo Playback
    ap.add_argument("--demo", type=str, default=None, help="Path to a JSONL log file to replay search results from.")
    return ap

def main():
    print("[H-MAPS] System Starting (Multi-threaded & Robust Regression)...", flush=True)
    parser = build_argparser()
    args = parser.parse_args()
    app = QApplication(sys.argv)
    
    signal.signal(signal.SIGINT, lambda s, f: app.quit())
    print("[INFO] Initializing GUI...", flush=True)
    overlay = demo_gui.Overlay()
    overlay.show()
    QTimer.singleShot(100, overlay._toggle_minimize) 
    controller = MainController(args, overlay)
    controller.start()
    end_time = time.time()
    print(f"[PERF] Startup Time: {end_time - start_time:.2f} seconds", flush=True)
    print("[H-MAPS] Main loop started.", flush=True)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()