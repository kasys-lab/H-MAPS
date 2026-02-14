#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
demo_backend.py
- Azure OpenAI クライアント設定 (推論・クエリ生成用)
- Local LLM (llama.cpp) 設定 (プライバシー保護要約・記憶更新用)
- S2ORC 検索 (Retriever)
- OCR Wrapper
- スクリーンキャプチャ関数
- 記憶管理 (Memory Manager)
"""

import os
import re
import hashlib
import sqlite3
import numpy as np
import mss
from PIL import Image
from pathlib import Path
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import time
from datetime import datetime
import json
import sys

# --- Local LLM Import ---
try:
    from llama_cpp import Llama
    HAS_LOCAL_LLM = True
except ImportError:
    HAS_LOCAL_LLM = False
    print("[WARN] llama-cpp-python not found. Local LLM features disabled.")


# OpenMP / tokenizers の警告対策
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ================= Azure OpenAI 設定 =================
from openai import AzureOpenAI

CANDIDATE_ENV_FILES = ["secrets/secrets.txt"]

DEMO_PERSONAS = {
    "nlp": """
User is a Computer Science researcher specializing in NLP algorithms and RAG architectures. Primary interests include inference latency optimization, model compression , and the computational complexity of reflection token generation. User prioritizes backend performance and system-level efficiency over user interface design.
""",
    "hci": """
User is an HCI researcher focusing on Human-AI collaboration and trust calibration. Primary interests include the visualization of citations , the impact of reflection tokens on user cognitive load, and interface transparency. User prioritizes explaining AI outputs over backend algorithmic details.
"""
}


def load_env_like_file(path: str) -> bool:
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = [p.strip() for p in line.split("=", 1)]
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    print(f"[info] Loaded secrets from {path}")
    return True

for env_file in CANDIDATE_ENV_FILES:
    if load_env_like_file(env_file):
        break

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT")

client = None
if endpoint and api_key:
    endpoint = endpoint.rstrip("/")
    try:
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        print(f"[INFO] Azure OpenAI client initialized. Endpoint: {endpoint}")
    except Exception as e:
        print(f"[WARN] Failed to init Azure OpenAI client: {e}")
else:
    print("[WARN] AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set.")


# ================= Local LLM 管理 =================

_local_llm = None

def init_local_llm(model_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096):
    """
    アプリケーション起動時に一度だけ呼ばれる初期化関数
    """
    global _local_llm
    if not HAS_LOCAL_LLM:
        print("[ERR] Cannot init local LLM: library not installed.")
        return

    if not os.path.exists(model_path):
        print(f"[ERR] Local model not found at: {model_path}")
        return

    print(f"[INFO] Loading Local LLM from {model_path} ...", flush=True)
    try:
        # n_gpu_layers=-1 でGPUフルオフロード
        _local_llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers, 
            n_ctx=n_ctx,
            verbose=False
        )
        print("[INFO] Local LLM loaded successfully.", flush=True)
    except Exception as e:
        print(f"[ERR] Failed to load Local LLM: {e}")
        _local_llm = None


# ================= LLM Helper (Local) =================

def _run_local_llm(prompt: str, max_tokens: int = 300) -> str:
    """ローカルLLM実行用ヘルパー"""
    if not _local_llm:
        print("[WARN] Local LLM is not initialized.")
        return ""
    
    try:
        output = _local_llm(
            prompt,
            max_tokens=max_tokens,
            stop=["<|end|>", "\n\n", "<|user|>"],
            temperature=0.1,
            echo=False
        )
        return output["choices"][0]["text"].strip()
    except Exception as e:
        print(f"[ERR] Local LLM inference failed: {e}")
        return ""


# ================= S2ORC 検索 =================

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if last_hidden_state.size(1) != attention_mask.size(1):
        L = min(last_hidden_state.size(1), attention_mask.size(1))
        last_hidden_state = last_hidden_state[:, :L, :]
        attention_mask = attention_mask[:, :L]
    mask = attention_mask.unsqueeze(-1)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-6)
    return summed / lengths

class S2ORCRetriever:
    def __init__(
        self,
        model_dir: str,
        index_path: str,
        ids_path: str,
        db_path: str,
        max_length: int = 512,
        top_k: int = 5,
    ):
        print("[INFO] S2ORCRetriever.__init__ start", flush=True)
        self.model_dir = model_dir
        self.index_root = Path(index_path)
        self.ids_root = Path(ids_path)
        self.db_path = db_path
        self.max_length = max_length
        self.top_k = top_k
        self.device = torch.device("cpu") # 強制CPU (GPUがある場合は "cuda" に変更可)

        print(f"[INFO] loading model from {self.model_dir}", flush=True)
        self.model = AutoModel.from_pretrained(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.use_e5_prefix = "e5" in self.model_dir.lower()

        print(f"[INFO] loading index/ids", flush=True)
        self._load_faiss_shards(self.index_root, self.ids_root)

        print(f"[INFO] connecting to DB: {self.db_path}", flush=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._prepare_db_columns()
        print("[INFO] S2ORCRetriever.__init__ done", flush=True)

    def _load_faiss_shards(self, index_root: Path, ids_root: Path):
        index_files = sorted(index_root.glob("*.index")) if index_root.is_dir() else [index_root]
        ids_files = sorted(ids_root.glob("*.npy")) if ids_root.is_dir() else [ids_root]

        if not index_files or not ids_files:
            raise RuntimeError("Index/IDs files not found.")

        n_pairs = min(len(index_files), len(ids_files))
        self.indexes: List[faiss.Index] = []
        self.ids_list: List[np.ndarray] = []

        total_vecs = 0
        for i in range(n_pairs):
            ip = index_files[i]
            npy = ids_files[i]
            print(f"[INFO] loading index shard {i}: {ip} ids: {npy}", flush=True)
            idx = faiss.read_index(str(ip))
            ids_arr = np.load(str(npy))
            total_vecs += idx.ntotal
            self.indexes.append(idx)
            self.ids_list.append(ids_arr)
        
        print(f"==================================================", flush=True)
        print(f"[INFO] S2ORC Index Loaded.", flush=True)
        print(f"[INFO] Total Vectors: {total_vecs:,} (approx. {total_vecs/1000000:.1f}M)", flush=True)
        print(f"==================================================", flush=True)

    def _prepare_db_columns(self):
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(papers)")
        cols = [row[1] for row in cur.fetchall()]
        self.title_col = "title" if "title" in cols else None
        self.abstract_col = "abstract" if "abstract" in cols else ("summary" if "summary" in cols else None)
        self.text_col = "text" if "text" in cols else None
        self.year_col = "year" if "year" in cols else None
        self.venue_col = "venue" if "venue" in cols else None
        self.url_col = next((c for c in ["url", "s2_url", "paper_url"] if c in cols), None)
        self.doi_col = "doi" if "doi" in cols else None

    def encode_query(self, text: str) -> np.ndarray:
        q = "query: " + text if self.use_e5_prefix else text
        with torch.no_grad():
            enc = self.tokenizer([q], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            for k in enc: enc[k] = enc[k].to(self.device)
            out = self.model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype("float32")

    def _fetch_paper_row(self, paper_id: int):
        cols = [c for c in [self.title_col, self.abstract_col, self.text_col, self.year_col, self.venue_col, self.url_col, self.doi_col] if c]
        select_cols = ", ".join(cols) if cols else "*"
        sql = f"SELECT {select_cols} FROM papers WHERE paper_id=?"
        cur = self.conn.cursor()
        cur.execute(sql, (paper_id,))
        return cur.fetchone(), cols

    def search(self, text: str, top_k: int | None = None) -> List[dict]:
        t_start = time.time()

        if top_k is None: top_k = self.top_k
        q_emb = self.encode_query(text)
        
        all_scores, all_pids = [], []
        for idx, ids_arr in zip(self.indexes, self.ids_list):
            D, I = idx.search(q_emb, top_k)
            for score, loc_idx in zip(D[0], I[0]):
                all_scores.append(float(score))
                all_pids.append(int(ids_arr[loc_idx]))
        
        if not all_scores: return []
        
        # Sort and Dedup
        all_scores = np.asarray(all_scores)
        all_pids = np.asarray(all_pids)
        order = np.argsort(-all_scores)
        seen, final_pids, final_scores = set(), [], []
        
        for idx in order:
            pid = int(all_pids[idx])
            if pid in seen: continue
            seen.add(pid)
            final_pids.append(pid)
            final_scores.append(float(all_scores[idx]))
            if len(final_pids) >= top_k: break
            
        results = []
        for pid, score in zip(final_pids, final_scores):
            row, cols = self._fetch_paper_row(pid)
            if not row:
                results.append({"paper_id": pid, "score": score, "title": f"[ID={pid} Not Found]", "abstract": "", "url": None})
                continue
                
            col_idx = {c: i for i, c in enumerate(cols)}
            def val(name, default=None): return row[col_idx[name]] if name in col_idx else default

            title = val(self.title_col, f"[ID={pid}]")
            abstract = val(self.abstract_col, "") or ""
            if not abstract and val(self.text_col): abstract = val(self.text_col)[:600]
            
            meta_parts = [str(x) for x in [val(self.year_col), val(self.venue_col)] if x]
            meta = " / ".join(meta_parts)
            
            url = val(self.url_col)
            doi = val(self.doi_col)
            if not url and doi: url = f"https://doi.org/{doi}"
            
            results.append({
                "paper_id": pid, "score": score, "title": title,
                "abstract": abstract[:800], "meta": meta, "url": url
            })
        t_end = time.time()
        print(f"[INFO] S2ORC Search took {t_end - t_start:.2f} seconds.")
        return results

# ================= OCR =================

class OCRWrapper:
    def __init__(self, lang: str = "eng"):
        import pyocr
        import pyocr.builders
        tools = pyocr.get_available_tools()
        if not tools: raise RuntimeError("No OCR tool found.")
        self._tool = tools[0]
        self._lang = lang
        self._builders = pyocr.builders
        print(f"[INFO] OCR initialized: {self._tool.get_name()}, lang={self._lang}")

    def image_to_text(self, image: Image.Image) -> str:
        return self._tool.image_to_string(
            image, lang=self._lang, builder=self._builders.TextBuilder()
        )

# ================= 画面キャプチャ =================

def capture_screen_image(mask_rect: Optional[tuple] = None) -> Image.Image:
    """
    mask_rect: (x, y, w, h) - この領域を白塗りする（オーバーレイ除外用）
    """
    with mss.mss() as sct:
        mon = sct.monitors[1]
        raw = sct.grab(mon)
        arr = np.array(raw) # BGRA

        if arr.shape[2] == 4:
            arr = arr[:, :, :3][:, :, ::-1]
        else:
            arr = arr[:, :, ::-1]

        if mask_rect:
            ox, oy, ow, oh = mask_rect
            x0 = ox - mon["left"]
            y0 = oy - mon["top"]
            H, W, _ = arr.shape
            x1 = max(0, min(W, x0 + ow))
            y1 = max(0, min(H, y0 + oh))
            x0 = max(0, min(W, x0))
            y0 = max(0, min(H, y0))

            if x1 > x0 and y1 > y0:
                arr[y0:y1, x0:x1] = 255 

        return Image.fromarray(arr)
    

# ================= 記憶管理 (Memory Manager) =================

class MemoryManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # ファイルパス定義
        self.path_sensory = self.base_dir / "sensory_stream.jsonl"
        self.path_local = self.base_dir / "local_context.jsonl"
        self.path_session = self.base_dir / "session_context.jsonl"
        self.path_profile = self.base_dir / "inferred_profile.json"

        self.path_search_log = self.base_dir / "search_log.jsonl"
        self.path_prompt_log = self.base_dir / "prompt_log.jsonl"

    def inject_demo_profile(self, persona_key: str):
        '''
        persona_key: 'nlp' or 'hci'
        '''
        # demo
        if persona_key not in DEMO_PERSONAS:
            print(f"[WARN] Unknown persona '{persona_key}'. Skipping injection.")
            return

        profile_text = DEMO_PERSONAS[persona_key]
        print(f"[DEMO] Injecting inferred Profile for: {persona_key.upper()}", flush=True)
        self.save_inferred_profile(profile_text)

    def save_search_log(self, payload: List[dict]):
        if not payload: return
        log_entry = {
            "timestamp": time.time(),
            "dt_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "events": []
        }
        for item in payload:
            question = item.get("question", "")
            cards = item.get("cards", [])
            paper_logs = []
            for c in cards:
                paper_logs.append({
                    "paper_id": c.get("paper_id"),
                    "title": c.get("title"),
                    "url": c.get("url"),
                    "summary": c.get("summary") 
                })
            log_entry["events"].append({"question": question, "papers": paper_logs})

        self._append_jsonl(self.path_search_log, log_entry)
        print(f"[LOG] Saved search results to {self.path_search_log}", flush=True)

    def save_prompt_log(self, prompt: str, trigger_type: str):
        path = self.base_dir / "prompt_log.jsonl"
        self._append_jsonl(path, {
            "trigger_type": trigger_type,
            "prompt": prompt
        })

    def _append_jsonl(self, path: Path, data: dict):
        data["timestamp"] = time.time()
        data["dt_str"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # --- 1. Sensory Stream ---
    def save_sensory(self, text: str, text_hash: str):
        self._append_jsonl(self.path_sensory, {
            "type": "sensory",
            "text": text,
            "hash": text_hash
        })

    # --- 2. Local Context ---
    def save_local_context(self, summary: str, source_text_hash: str):
        self._append_jsonl(self.path_local, {
            "type": "local_context",
            "summary": summary,
            "source_hash": source_text_hash
        })

    def get_recent_local_contexts(self, limit: int = 5) -> List[str]:
        if not self.path_local.exists():
            return []
        lines = []
        with open(self.path_local, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line))
        return [item["summary"] for item in lines[-limit:]]

    # --- 3. Session Context ---
    def save_session_context(self, summary: str):
        self._append_jsonl(self.path_session, {
            "type": "session_context",
            "summary": summary
        })

    def get_latest_session_context(self) -> str:
        if not self.path_session.exists():
            return ""
        last_line = ""
        with open(self.path_session, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if last_line:
            return json.loads(last_line).get("summary", "")
        return ""

    # --- 4. inferred Profile ---
    def save_inferred_profile(self, profile_text: str):
        data = {
            "type": "inferred_profile",
            "profile": profile_text,
            "timestamp": time.time(),
            "dt_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.path_profile, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_inferred_profile(self) -> str:
        if not self.path_profile.exists():
            return "No profile available."
        try:
            with open(self.path_profile, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("profile", "")
        except:
            return ""

# ================= LLM Context Operations =================

def llm_summarize_chunk(text: str) -> str:
    """Step 2: Raw Chunk -> Local Context (Micro Summary) by Local LLM"""
    if _local_llm:
        prompt = f"""<|system|>
You are a research assistant. Summarize the following text concisely.
- Focus on technical keywords and definitions.
- Keep it under 3 sentences.
<|end|>
<|user|>
TEXT:
{text[:2500]}
<|end|>
<|assistant|>"""
        return _run_local_llm(prompt) or (text[:200] + "...")
    
    return text[:200].replace("\n", " ") + "..."

def llm_integrate_session(local_contexts: List[str]) -> str:
    """Step 3: Local Contexts -> Session Context """
    joined = "\n".join([f"- {s}" for s in local_contexts])
    
    # ローカルLLMがあるならローカルで処理（プライバシー優先）
    if _local_llm:
        prompt = f"""<|system|>
Summarize the current task goal based on the recent reading history.
Output 1-2 sentences describing "what the user is working on".
<|end|>
<|user|>
HISTORY:
{joined}
<|end|>
<|assistant|>"""
        res = _run_local_llm(prompt, max_tokens=150)
        if res: return res
        
    # フォールバック: Cloud (もしLocal LLMがない場合)
    if client:
        try:
            prompt = f"Summarize the user's current task based on these history items:\n{joined}"
            res = client.chat.completions.create(
                model=deployment_id, messages=[{"role": "user", "content": prompt}], max_tokens=200
            )
            return res.choices[0].message.content.strip()
        except: pass

    return "Processing..."

def llm_update_profile(current_profile: str, session_summary: str) -> str:
    """Step 4: Session -> inferred Profile"""
    
    # ローカルLLMでプロファイル更新を試みる
    if _local_llm:
        prompt = f"""<|system|>
Update the user's research profile. Incorporate the new session activity.
If new activity contradicts old profile, prioritize the new one (Concept Drift).
<|end|>
<|user|>
OLD PROFILE:
{current_profile}

NEW ACTIVITY:
{session_summary}
<|end|>
<|assistant|>"""
        res = _run_local_llm(prompt, max_tokens=300)
        if res: return res

    # フォールバック: Cloud
    if client:
        prompt = f"""Update User Profile.
OLD: {current_profile}
NEW ACTIVITY: {session_summary}
OUTPUT UPDATED PROFILE ONLY."""
        try:
            res = client.chat.completions.create(
                model=deployment_id, messages=[{"role": "user", "content": prompt}], max_tokens=300
            )
            return res.choices[0].message.content.strip()
        except: pass
        
    return current_profile

def generate_query_with_hmaps(
    current_text: str,
    local_context: List[str],
    session_context: str,
    inferred_profile: str,
    trigger_type: str = "expansion",
    provider: str = "openai" 
) -> tuple[List[str], str]:
    """
    Step 5: Query Generation
    - provider="openai": Uses Azure OpenAI (GPT-4o)
    - provider="local": Uses Local LLM (Phi-3 etc.) via _run_local_llm
    """
    t_gen_start = time.time()

    # --- Helper: Output Parser ---
    def _parse_questions(raw_text: str) -> List[str]:
        lines = []
        # 行ごとに分割し、'?'が含まれる長めの行を質問として採用
        # 番号付きリスト(1. )や箇条書き(- )を除去してクリーンにする
        for line in raw_text.splitlines():
            # 先頭の数字や記号を削除
            s = re.sub(r"^[\d\-\.\s]+", "", line.strip())
            if len(s) > 10 and "?" in s:
                lines.append(s)
        return lines[:3]

    lc_str = "\n".join(local_context)
    
    # --- Prompt Construction ---
    
    # 共通のシステムロール
    system_inst = "You are a research assistant."

    # トリガータイプごとの戦略記述
    if trigger_type == "support":
        # Clarification (Content Revisit)
        mode_inst = (
            "The user is currently scrolling back to reread the [Current Text] section, indicating a need to verify information or reconstruct context. "
            "Generate a 'Clarification-oriented' question to resolve the user's potential confusion. "
            "Using the [Inferred Profile] (long-term interest), [Session Context], and [Local Context], formulate a question that seeks literature providing "
            "standard definitions, structured explanations of procedures, or empirical comparisons relevant to the current text. "
            "Always output in the form of a natural language question with a question mark(?) at the end."
        )
    else:
        # Exploration (Sustained Attention) - default
        mode_inst = (
            "The user is currently reading the [Current Text] section with sustained attention, indicating deep engagement. "
            "Generate an 'Exploration-oriented' question to broaden the scope of the topic. "
            "Using the [Inferred Profile] (long-term interest), [Session Context], and [Local Context], formulate a question that inquires about "
            "related work, alternative methodologies, critical limitations, or comparison criteria relevant to the concept being read. "
            "Always output in the form of a natural language question with a question mark(?) at the end."
        )

    # データブロックの作成
    data_block = f"""
[Inferred Profile]
{inferred_profile}

[Session Context]
{session_context}

[Local Context]
{lc_str}

[Current Text]
{current_text}
"""

    generated_content = ""
    used_prompt = ""

    # === 分岐処理 ===

    # (A) Local LLM Execution
    if provider == "local":
        if not _local_llm:
            print("[WARN] Local LLM requested but not loaded. Falling back to simple text.", flush=True)
            return ([current_text], "Local LLM Not Loaded")
        
        # Phi-3 / Llama-3 向けのプロンプト形式 (ChatML style)
        prompt = f"""<|system|>
{system_inst}
{mode_inst}
Output 3 distinct questions. Output ONLY the questions one per line.
<|end|>
<|user|>
CONTEXT_DATA:
{data_block}

OUTPUT (3 queries):
<|end|>
<|assistant|>"""
        
        used_prompt = prompt
        # ローカルLLM実行
        generated_content = _run_local_llm(prompt, max_tokens=256)

    # (B) Azure OpenAI Execution (Default)
    else:
        if not client: 
            return ([current_text], "Client Not Configured")

        # OpenAI向けのメッセージ形式
        prompt_content = f"""{mode_inst}

{data_block}

OUTPUT (3 queries):
"""
        used_prompt = prompt_content
        try:
            res = client.chat.completions.create(
                model=deployment_id,
                messages=[
                    {"role": "system", "content": system_inst},
                    {"role": "user", "content": prompt_content}
                ],
                max_tokens=200
            )
            generated_content = res.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] Query gen failed: {e}", flush=True)
            return ([current_text], "")

    # === 共通の後処理 ===
    t_gen_end = time.time()
    print(f"[INFO] Query generation ({provider}) took {t_gen_end - t_gen_start:.2f} seconds.", flush=True)
    
    questions = _parse_questions(generated_content)
    
    # 質問生成に失敗した場合のフォールバック
    if not questions:
        print(f"[WARN] No valid questions parsed. Raw output: {generated_content[:50]}...", flush=True)
        # とりあえずCurrent Textの先頭を質問代わりにする
        fallback = current_text.split('\n')[0][:50]
        if "?" not in fallback: fallback += "?"
        questions = [fallback]

    return questions, used_prompt