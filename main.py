import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
import warnings
import curses
from collections import deque
from typing import Annotated
from typing_extensions import TypedDict

warnings.filterwarnings("ignore", message="Core Pydantic V1", module="pydantic")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AIMessageChunk, ToolMessage, RemoveMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_SETTINGS = {
    "agent": {
        "address": "http://100.66.64.45:9090/v1",
        "system_prompt": "",
        "cwd": "",
    },
    "classifier": {
        "address": "http://100.66.64.45:9091/v1",
        "system_prompt": "",
    },
}


def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
        result = json.loads(json.dumps(DEFAULT_SETTINGS))
        for section in ("agent", "classifier"):
            if section in data:
                result[section].update(data[section])
        return result
    except (FileNotFoundError, json.JSONDecodeError):
        return json.loads(json.dumps(DEFAULT_SETTINGS))


def save_settings(settings: dict) -> None:
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


SETTINGS: dict = load_settings()

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

_llm_cache: dict[tuple, ChatOpenAI] = {}


def get_llm(address: str, streaming: bool = True, json_mode: bool = False) -> ChatOpenAI:
    key = (address, streaming, json_mode)
    if key not in _llm_cache:
        kwargs: dict = {
            "base_url": address,
            "api_key": "not-needed",
            "model": "local-model",
            "streaming": streaming,
            "timeout": 120,  # prevent infinite hang when SSE stream stalls mid-response
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        _llm_cache[key] = ChatOpenAI(**kwargs)
    return _llm_cache[key]


def rebuild_llms() -> None:
    _llm_cache.clear()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    messages:     Annotated[list, add_messages]
    intent:       str
    domain:       str
    confidence:   float
    rag_needed:   bool
    tools_needed: list[str]
    routing_note: str
    summary:      str

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM = """\
You are a JSON classifier. Your ONLY output is a single JSON object — nothing else.
Do NOT greet the user. Do NOT explain. Do NOT use markdown. Do NOT add any text before or after the JSON.

Classify the user message with these exact fields:

intent: one of: chat, question, task, research, code, troubleshoot, document
domain: one of: general, coding, network, windows, hotel_it, verifone, ai_runtime, finance, legal_hr
confidence: number from 0.0 to 1.0
rag_needed: true or false
tools_needed: array of strings (empty array if none)

Your entire response must be exactly one JSON object like this:
{"intent":"troubleshoot","domain":"windows","confidence":0.92,"rag_needed":false,"tools_needed":[]}

Start your response with { and end with }. No other characters allowed."""

DOMAIN_PROMPTS = {
    "hotel_it":   "You are an expert Hotel IT support engineer.",
    "coding":     "You are an expert software engineer.",
    "network":    "You are an expert network engineer.",
    "windows":    "You are an expert Windows systems administrator.",
    "verifone":   "You are an expert Verifone payment systems technician.",
    "ai_runtime": "You are an expert AI infrastructure and runtime engineer.",
    "finance":    "You are a knowledgeable finance assistant.",
    "legal_hr":   "You are a knowledgeable HR and legal information assistant.",
    "general":    "You are a helpful general-purpose assistant.",
}

CONFIDENCE_THRESHOLD  = 0.65
WINDOW_SIZE           = 20
SUMMARIZE_THRESHOLD   = 40

SUMMARIZE_PROMPT = """\
Summarize the key facts and context from the following conversation history.
Focus on: names, IP addresses, hostnames, error messages, resolved issues, and ongoing tasks.
Be concise — this summary will be prepended to future responses to preserve context."""

CLASSIFIER_RETRY_MSG = (
    "WRONG. That output could not be parsed as JSON. "
    "Your ENTIRE response must be ONE JSON object — nothing before it, nothing after it. "
    "No prose. No markdown. No explanation. START WITH { END WITH }. Try again:"
)

CLASSIFIER_CONFIDENCE_RETRY_MSG = (
    "WRONG. Your confidence field is 0.0 — this is never a valid value. "
    "Confidence must be between 0.1 and 1.0 and reflect how certain you are of the classification. "
    "If genuinely unsure, use 0.5. Output the corrected JSON object now:"
)

MONITOR_URL = "http://100.66.64.45:8086/api/sakura/monitor"
LOGS_URL    = "http://100.66.64.45:8086/api/sakura/logs"

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_CREATE_NEW_PROCESS_GROUP = 0x00000200
_DETACHED_PROCESS         = 0x00000008


def _run_proc(cmd: list[str], cwd=None, timeout: int = 15) -> tuple[str, str]:
    """Run a subprocess with timeout, killing the full process tree if it stalls."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        creationflags=_CREATE_NEW_PROCESS_GROUP,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return out.strip(), err.strip()
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, timeout=5,
        )
        try:
            out, err = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        note = "[Killed after timeout — use launch_app for long-running processes]"
        return out.strip(), (err.strip() + "\n" + note).strip()


@tool
def run_powershell(command: str) -> str:
    """Execute a short-lived PowerShell command and return stdout/stderr (15s timeout).
    For network ping tests use ping.exe directly (e.g. 'ping.exe -n 4 192.168.1.1'), not Test-Connection.
    Do NOT use this to start GUI apps or servers — use launch_app instead."""
    cwd = SETTINGS.get("agent", {}).get("cwd", "").strip() or None
    out, err = _run_proc(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=cwd,
    )
    if err:
        return f"STDOUT:\n{out}\nSTDERR:\n{err}" if out else f"STDERR:\n{err}"
    return out or "(no output)"


@tool
def run_python(code: str) -> str:
    """Execute a short-lived Python snippet and return stdout/stderr (15s timeout).
    Do NOT use this to start GUI apps or long-running scripts — use launch_app instead."""
    cwd = SETTINGS.get("agent", {}).get("cwd", "").strip() or None
    out, err = _run_proc([sys.executable, "-c", code], cwd=cwd)
    if err:
        return f"STDOUT:\n{out}\nSTDERR:\n{err}" if out else f"STDERR:\n{err}"
    return out or "(no output)"


@tool
def launch_app(command: str) -> str:
    """Launch a long-running application (GUI app, server, background script) and return immediately.
    The process is fully detached — use this instead of run_powershell/run_python when you want
    to START something without waiting for it to finish (e.g. 'python main.py', 'npm start')."""
    cwd = SETTINGS.get("agent", {}).get("cwd", "").strip() or None
    try:
        flags = _CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS
        subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            creationflags=flags,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        return f"Launched (detached): {command}"
    except Exception as exc:
        return f"[Error: {exc}]"


TOOLS    = [run_powershell, run_python, launch_app]
TOOL_MAP = {t.name: t for t in TOOLS}


def _parse_text_tool_calls(content: str) -> list[dict]:
    """Parse <tool_call> XML blocks emitted by models that don't use structured function calling."""
    calls = []
    for block in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        fn_match = re.search(r"<function=(\w+)>", block.group(1))
        if not fn_match:
            continue
        name   = fn_match.group(1)
        params = {}
        for p in re.finditer(r"<parameter=(\w+)>(.*?)</parameter>", block.group(1), re.DOTALL):
            params[p.group(1)] = p.group(2).strip()
        calls.append({"name": name, "args": params})
    return calls


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response that may include prose or code fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        raw = re.search(r"\{.*\}", text, re.DOTALL)
        if raw:
            text = raw.group(0)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def classify(state: AgentState) -> dict:
    cfg        = SETTINGS["classifier"]
    sys_prompt = cfg["system_prompt"] or CLASSIFIER_SYSTEM
    llm        = get_llm(cfg["address"], streaming=False, json_mode=True)

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        state["messages"][-1],
    )

    raw  = ""
    data = {}
    for attempt in range(2):
        if attempt == 0:
            messages = [SystemMessage(content=sys_prompt), last_human]
        else:
            messages = [
                SystemMessage(content=sys_prompt),
                last_human,
                AIMessage(content=raw),
                HumanMessage(content=CLASSIFIER_RETRY_MSG),
            ]

        response = llm.invoke(messages)
        raw  = response.content if isinstance(response.content, str) else ""
        data = _extract_json(raw)
        if data:
            break

    retried = attempt > 0
    if not data:
        snippet = raw[:120].replace("\n", " ") if raw else "<empty>"
        return {
            "intent": "chat", "domain": "general", "confidence": 0.0,
            "rag_needed": False, "tools_needed": [],
            "routing_note": f"[Router → parse failed (2 attempts) | raw: {snippet}]",
        }

    # Shock retry: if confidence came back 0.0, demand a real value
    shocked = False
    if float(data.get("confidence", 0.0)) <= 0.0:
        shock_resp = llm.invoke([
            SystemMessage(content=sys_prompt),
            last_human,
            AIMessage(content=raw),
            HumanMessage(content=CLASSIFIER_CONFIDENCE_RETRY_MSG),
        ])
        shock_raw  = shock_resp.content if isinstance(shock_resp.content, str) else ""
        shock_data = _extract_json(shock_raw)
        if shock_data and float(shock_data.get("confidence", 0.0)) > 0.0:
            data = shock_data
            raw  = shock_raw
        shocked = True

    intent       = str(data.get("intent", "chat"))
    domain       = str(data.get("domain", "general"))
    confidence   = float(data.get("confidence", 0.0))
    rag_needed   = bool(data.get("rag_needed", False))
    tools_needed = list(data.get("tools_needed", []))

    tools_str  = ", ".join(tools_needed) if tools_needed else "none"
    rag_str    = " | RAG" if rag_needed else ""
    retry_str  = " | retried" if retried else ""
    shock_str  = " ⚡ zero-conf" if (shocked and confidence <= 0.0) else ""
    routing_note = (
        f"[Router → {intent}/{domain}"
        f" (conf: {confidence:.2f}){rag_str}{shock_str}"
        f" | tools: {tools_str}{retry_str}]"
    )

    return {
        "intent": intent, "domain": domain, "confidence": confidence,
        "rag_needed": rag_needed, "tools_needed": tools_needed,
        "routing_note": routing_note,
    }


def route(state: AgentState) -> str:
    conf = state.get("confidence", 0.0)
    if conf <= 0.0:
        return "respond"  # classifier failed completely — just answer
    if conf < CONFIDENCE_THRESHOLD:
        return "clarify"
    if state.get("rag_needed", False):
        return "rag"
    return "respond"


def _build_sys_prompt(base: str, state: AgentState) -> str:
    parts = [base]
    cwd = SETTINGS.get("agent", {}).get("cwd", "").strip()
    if cwd:
        parts.append(f"Current working directory: {cwd}")
    summary = state.get("summary", "")
    if summary:
        parts.append(f"Earlier conversation summary:\n{summary}")
    return "\n\n".join(parts)


def clarify(state: AgentState) -> dict:
    cfg      = SETTINGS["agent"]
    base     = (
        "The user's request is unclear. "
        "Ask one short clarifying question to better understand what they need. "
        "Do not answer the question itself."
    )
    response = get_llm(cfg["address"]).invoke([
        SystemMessage(content=_build_sys_prompt(base, state)),
        *state["messages"][-WINDOW_SIZE:],
    ])
    conf = state.get("confidence", 0.0)
    return {
        "messages":     [AIMessage(content=response.content)],
        "routing_note": f"[Router → low confidence ({conf:.2f}) — asking for clarification]",
    }


def rag(state: AgentState) -> dict:
    return {"routing_note": state.get("routing_note", "") + " [RAG: pending]"}


def respond(state: AgentState) -> dict:
    cfg           = SETTINGS["agent"]
    domain        = state.get("domain", "general")
    domain_prompt = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
    custom        = cfg["system_prompt"]
    base          = (custom + "\n\n" + domain_prompt) if custom else domain_prompt

    llm      = get_llm(cfg["address"]).bind_tools(TOOLS)
    response = llm.invoke([
        SystemMessage(content=_build_sys_prompt(base, state)),
        *state["messages"][-WINDOW_SIZE:],
    ])
    return {"messages": [response]}


def summarize(state: AgentState) -> dict:
    messages = state["messages"]
    if len(messages) <= SUMMARIZE_THRESHOLD:
        return {}

    to_compress = messages[:-WINDOW_SIZE]
    cfg         = SETTINGS["agent"]
    existing    = state.get("summary", "")
    prefix      = f"Previous summary:\n{existing}\n\nNew messages to incorporate:\n" if existing else ""
    history_text = "\n".join(
        f"{m.__class__.__name__}: {getattr(m, 'content', '')}"
        for m in to_compress
    )

    response = get_llm(cfg["address"], streaming=False).invoke([
        SystemMessage(content=SUMMARIZE_PROMPT),
        HumanMessage(content=prefix + history_text),
    ])
    new_summary = response.content if isinstance(response.content, str) else ""

    return {
        "summary":  new_summary,
        "messages": [RemoveMessage(id=m.id) for m in to_compress],
    }


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def dispatch_text_tools(state: AgentState) -> dict:
    """Execute tool calls embedded as <tool_call> XML in the AI's text response."""
    last    = state["messages"][-1]
    content = getattr(last, "content", "") or ""
    calls   = _parse_text_tool_calls(content)
    results = []
    for call in calls:
        fn = TOOL_MAP.get(call["name"])
        if fn:
            try:
                output = fn.invoke(call["args"])
            except Exception as exc:
                output = f"[Error: {exc}]"
            results.append(ToolMessage(
                content=str(output),
                name=call["name"],
                tool_call_id=call["name"],
            ))
    return {"messages": results}


def _after_respond(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    if "<tool_call>" in (getattr(last, "content", "") or ""):
        return "dispatch_text_tools"
    return "summarize"


builder = StateGraph(AgentState)
builder.add_node("classify", classify)
builder.add_node("clarify",  clarify)
builder.add_node("rag",      rag)
builder.add_node("respond",              respond)
builder.add_node("tools",               ToolNode(TOOLS))
builder.add_node("dispatch_text_tools", dispatch_text_tools)
builder.add_node("summarize",           summarize)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route, {
    "clarify": "clarify",
    "rag":     "rag",
    "respond": "respond",
})
builder.add_edge("rag",     "respond")
builder.add_edge("clarify", END)
builder.add_conditional_edges("respond", _after_respond, {
    "tools":               "tools",
    "dispatch_text_tools": "dispatch_text_tools",
    "summarize":           "summarize",
})
builder.add_edge("tools",               "respond")
builder.add_edge("dispatch_text_tools", "respond")
builder.add_edge("summarize",           END)

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------

MENU       = [("F1", "Chat"), ("F12", "Settings")]
VIEW_HOME  = "home"
VIEW_CHAT  = "chat"
VIEW_SETTINGS = "settings"

ROLE_PAIR   = {"user": 2, "ai": 3, "router": 4, "tool": 5}
ROLE_PREFIX = {"user": "You: ", "ai": "AI:  ", "router": "", "tool": ""}

F_AGENT_ADDR        = 0
F_AGENT_PROMPT      = 1
F_AGENT_CWD         = 2
F_CLASSIFIER_ADDR   = 3
F_CLASSIFIER_PROMPT = 4
NUM_FIELDS          = 5

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class App:
    def __init__(self, stdscr, graph):
        self.stdscr    = stdscr
        self.graph     = graph
        self.view      = VIEW_HOME
        self.prev_view = VIEW_HOME
        self.thread_id = "default"
        self.history   = []
        self.input_buf    = ""
        self.input_cursor = 0
        self.thinking     = False
        self.chat_scroll  = 0
        self._monitor_data: dict | list = {}
        self._gpu_history: deque = deque(maxlen=20)  # 20 × 0.5s = 10s rolling window
        self._log_lines: list[str] = []
        self._stop_event    = threading.Event()
        self._stream_queue  = queue.Queue()
        self._cancel_event  = threading.Event()
        self._ai_idx: int | None = None
        self._last_queue_event: float = 0.0
        # Settings editor state
        self.settings_focus   = 0
        self.settings_bufs    = self._bufs_from_settings()
        self.settings_cursors = [len(b) for b in self.settings_bufs]

    def _bufs_from_settings(self) -> list[str]:
        return [
            SETTINGS["agent"]["address"],
            SETTINGS["agent"]["system_prompt"],
            SETTINGS["agent"].get("cwd", ""),
            SETTINGS["classifier"]["address"],
            SETTINGS["classifier"]["system_prompt"],
        ]

    def run(self):
        curses.curs_set(1)
        self.stdscr.keypad(True)
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)

        if curses.can_change_color() and curses.COLORS >= 16:
            curses.init_color(8,  1000, 714, 796)
            curses.init_color(9,   471, 235, 706)
            curses.init_color(10,  961, 663, 722)
            curses.init_color(11,  357, 808, 980)
            curses.init_color(12,  784, 667, 902)
            curses.init_color(13,  588, 902, 392)   # pastel lime-green
            pink, purple = 8, 9
            trans_pink, trans_blue, pastel_purple, lime = 10, 11, 12, 13
        else:
            pink, purple       = curses.COLOR_MAGENTA, curses.COLOR_BLUE
            trans_pink         = curses.COLOR_MAGENTA
            trans_blue         = curses.COLOR_CYAN
            pastel_purple      = curses.COLOR_MAGENTA
            lime               = curses.COLOR_GREEN

        curses.init_pair(1, purple,        pink)
        curses.init_pair(2, trans_pink,    curses.COLOR_BLACK)
        curses.init_pair(3, trans_blue,    curses.COLOR_BLACK)
        curses.init_pair(4, pastel_purple, curses.COLOR_BLACK)
        curses.init_pair(5, lime,          curses.COLOR_BLACK)

        # 1-second getch timeout so the monitor panel refreshes without input
        self.stdscr.timeout(1000)
        self._start_monitor_thread()

        while True:
            self.stdscr.erase()
            self._draw_menubar()
            if self.view == VIEW_CHAT:
                self._draw_chat()
            elif self.view == VIEW_SETTINGS:
                self._draw_settings()
            else:
                self._draw_home()
            self.stdscr.refresh()

            key = self.stdscr.getch()
            if key == curses.ERR:
                self._drain_queue()
                continue
            if not self._handle_key(key):
                break
            self._drain_queue()

        self._stop_event.set()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _handle_key(self, key) -> bool:
        if key == curses.KEY_F1:
            self.view = VIEW_CHAT
        elif key == curses.KEY_F12:
            self.prev_view        = self.view
            self.settings_bufs    = self._bufs_from_settings()
            self.settings_cursors = [len(b) for b in self.settings_bufs]
            self.settings_focus   = 0
            self.view             = VIEW_SETTINGS
        elif key == 27:  # ESC
            if self.thinking:
                self._cancel_event.set()
            elif self.view == VIEW_SETTINGS:
                self._commit_settings()
                self.view = self.prev_view
            elif self.view == VIEW_HOME:
                return False
            else:
                self.view = VIEW_HOME
        elif key == curses.KEY_MOUSE:
            try:
                _, _, _, _, bstate = curses.getmouse()
                if self.view == VIEW_CHAT:
                    if bstate & curses.BUTTON4_PRESSED:
                        self.chat_scroll = max(0, self.chat_scroll + 3)
                    btn5 = getattr(curses, 'BUTTON5_PRESSED', 0)
                    if btn5 and bstate & btn5:
                        self.chat_scroll = max(0, self.chat_scroll - 3)
            except curses.error:
                pass
        elif self.view == VIEW_SETTINGS:
            self._handle_settings_key(key)
        elif self.view == VIEW_CHAT:
            h, w = self.stdscr.getmaxyx()
            if key == curses.KEY_PPAGE:
                self.chat_scroll = max(0, self.chat_scroll + h // 2)
            elif key == curses.KEY_NPAGE:
                self.chat_scroll = max(0, self.chat_scroll - h // 2)
            elif not self.thinking:
                field_w = w - 3
                pos     = self.input_cursor
                if key in (curses.KEY_ENTER, 10, 13):
                    # Peek ahead: if more input is queued, this newline came from paste — insert it.
                    # If nothing is waiting, the user pressed Enter — send.
                    self.stdscr.timeout(0)
                    lookahead = self.stdscr.getch()
                    self.stdscr.timeout(1000)
                    if lookahead != curses.ERR:
                        curses.ungetch(lookahead)
                        self.input_buf    = self.input_buf[:pos] + "\n" + self.input_buf[pos:]
                        self.input_cursor = pos + 1
                    else:
                        self._send()
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    if pos > 0:
                        self.input_buf    = self.input_buf[:pos - 1] + self.input_buf[pos:]
                        self.input_cursor = pos - 1
                elif key == curses.KEY_DC:
                    if pos < len(self.input_buf):
                        self.input_buf = self.input_buf[:pos] + self.input_buf[pos + 1:]
                elif key == curses.KEY_LEFT:
                    self.input_cursor = max(0, pos - 1)
                elif key == curses.KEY_RIGHT:
                    self.input_cursor = min(len(self.input_buf), pos + 1)
                elif key == curses.KEY_HOME:
                    self.input_cursor = 0
                elif key == curses.KEY_END:
                    self.input_cursor = len(self.input_buf)
                elif key == curses.KEY_UP:
                    vlines = self._compute_visual_lines(self.input_buf, field_w)
                    row, col = self._cursor_to_visual(vlines, pos)
                    if row > 0:
                        start, line = vlines[row - 1]
                        self.input_cursor = start + min(col, len(line))
                elif key == curses.KEY_DOWN:
                    vlines = self._compute_visual_lines(self.input_buf, field_w)
                    row, col = self._cursor_to_visual(vlines, pos)
                    if row < len(vlines) - 1:
                        start, line = vlines[row + 1]
                        self.input_cursor = start + min(col, len(line))
                elif 32 <= key <= 126:
                    self.input_buf    = self.input_buf[:pos] + chr(key) + self.input_buf[pos:]
                    self.input_cursor = pos + 1
        return True

    def _handle_settings_key(self, key) -> None:
        idx       = self.settings_focus
        buf       = self.settings_bufs[idx]
        pos       = self.settings_cursors[idx]
        is_prompt = idx in (F_AGENT_PROMPT, F_CLASSIFIER_PROMPT)

        _, w    = self.stdscr.getmaxyx()
        field_w = max(10, w - 18 - 4)

        if key == 9:  # Tab → next field
            self.settings_focus = (idx + 1) % NUM_FIELDS
        elif key == curses.KEY_UP:
            if is_prompt:
                vlines = self._compute_visual_lines(buf, field_w)
                row, col = self._cursor_to_visual(vlines, pos)
                if row > 0:
                    start, line = vlines[row - 1]
                    self.settings_cursors[idx] = start + min(col, len(line))
                else:
                    self.settings_focus = (idx - 1) % NUM_FIELDS
            else:
                self.settings_focus = (idx - 1) % NUM_FIELDS
        elif key == curses.KEY_DOWN:
            if is_prompt:
                vlines = self._compute_visual_lines(buf, field_w)
                row, col = self._cursor_to_visual(vlines, pos)
                if row < len(vlines) - 1:
                    start, line = vlines[row + 1]
                    self.settings_cursors[idx] = start + min(col, len(line))
                else:
                    self.settings_focus = (idx + 1) % NUM_FIELDS
            else:
                self.settings_focus = (idx + 1) % NUM_FIELDS
        elif key == curses.KEY_LEFT:
            self.settings_cursors[idx] = max(0, pos - 1)
        elif key == curses.KEY_RIGHT:
            self.settings_cursors[idx] = min(len(buf), pos + 1)
        elif key == curses.KEY_HOME:
            self.settings_cursors[idx] = 0
        elif key == curses.KEY_END:
            self.settings_cursors[idx] = len(buf)
        elif key in (curses.KEY_BACKSPACE, 127, 8):
            if pos > 0:
                self.settings_bufs[idx]    = buf[:pos - 1] + buf[pos:]
                self.settings_cursors[idx] = pos - 1
        elif key == curses.KEY_DC:
            if pos < len(buf):
                self.settings_bufs[idx] = buf[:pos] + buf[pos + 1:]
        elif key in (curses.KEY_ENTER, 10, 13):
            if is_prompt:
                self.settings_bufs[idx]    = buf[:pos] + "\n" + buf[pos:]
                self.settings_cursors[idx] = pos + 1
            else:
                self.settings_focus = (idx + 1) % NUM_FIELDS
        elif 32 <= key <= 126:
            self.settings_bufs[idx]    = buf[:pos] + chr(key) + buf[pos:]
            self.settings_cursors[idx] = pos + 1

    def _commit_settings(self) -> None:
        SETTINGS["agent"]["address"]            = self.settings_bufs[F_AGENT_ADDR]
        SETTINGS["agent"]["system_prompt"]      = self.settings_bufs[F_AGENT_PROMPT]
        SETTINGS["agent"]["cwd"]               = self.settings_bufs[F_AGENT_CWD]
        SETTINGS["classifier"]["address"]       = self.settings_bufs[F_CLASSIFIER_ADDR]
        SETTINGS["classifier"]["system_prompt"] = self.settings_bufs[F_CLASSIFIER_PROMPT]
        save_settings(SETTINGS)
        rebuild_llms()

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        self.chat_scroll = 0
        self.stdscr.erase()
        self._draw_menubar()
        self._draw_chat()
        self.stdscr.refresh()

    def _send(self):
        text = self.input_buf.strip()
        if not text:
            return
        self.input_buf    = ""
        self.input_cursor = 0
        self.chat_scroll  = 0
        self.history.append(("user", text))
        self.thinking  = True
        self._log_lines = []
        self._ai_idx    = None
        self._last_queue_event = time.monotonic()
        self._cancel_event.clear()
        self._start_log_thread()
        self._redraw()
        threading.Thread(target=self._stream_worker, args=(text,), daemon=True).start()
        threading.Thread(target=self._stream_watchdog, daemon=True).start()

    def _stream_worker(self, text: str) -> None:
        q = self._stream_queue
        try:
            pending_ai = False
            for item in self.graph.stream(
                {"messages": [HumanMessage(content=text)]},
                config={"configurable": {"thread_id": self.thread_id}},
                stream_mode=["updates", "messages"],
            ):
                if self._cancel_event.is_set():
                    break
                mode, data = item

                if mode == "updates":
                    for node_name, update in data.items():
                        if node_name == "classify" and update.get("routing_note"):
                            q.put(("classify", update["routing_note"]))
                        elif node_name == "tools":
                            for msg in update.get("messages", []):
                                q.put(("tool", getattr(msg, "name", "tool"), getattr(msg, "content", "")))
                            q.put(("ai_next",))
                            pending_ai = False
                        elif node_name == "dispatch_text_tools":
                            q.put(("ai_strip_tool_calls",))
                            for msg in update.get("messages", []):
                                q.put(("tool", getattr(msg, "name", "tool"), getattr(msg, "content", "")))
                            q.put(("ai_next",))
                            pending_ai = False

                elif mode == "messages":
                    msg_chunk, metadata = data
                    if not isinstance(msg_chunk, AIMessageChunk):
                        continue
                    content = getattr(msg_chunk, "content", "") or ""
                    if metadata.get("langgraph_node", "") in ("respond", "clarify") and content:
                        q.put(("ai_start", content) if not pending_ai else ("ai_append", content))
                        pending_ai = True

        except Exception as exc:
            q.put(("error", str(exc)))
            return
        q.put(("done",))

    def _stream_watchdog(self) -> None:
        """If no queue activity for 60 s while thinking, the stream is stalled — force-complete."""
        while self.thinking:
            time.sleep(2)
            if self.thinking and (time.monotonic() - self._last_queue_event) > 60:
                self._stream_queue.put(("error", "Stream stalled — no activity for 60s"))
                return

    def _drain_queue(self) -> None:
        needs_redraw = False
        while True:
            try:
                event = self._stream_queue.get_nowait()
            except queue.Empty:
                break
            self._last_queue_event = time.monotonic()
            kind = event[0]
            if kind == "classify":
                self.history.append(("router", event[1]))
                needs_redraw = True
            elif kind == "tool":
                self.history.append(("tool", f"[Tool: {event[1]}]\n{event[2]}"))
                self._ai_idx = None
                needs_redraw = True
            elif kind == "ai_start":
                self.history.append(("ai", event[1]))
                self._ai_idx = len(self.history) - 1
                needs_redraw = True
            elif kind == "ai_append":
                if self._ai_idx is not None:
                    role, prev = self.history[self._ai_idx]
                    self.history[self._ai_idx] = (role, prev + event[1])
                    needs_redraw = True
            elif kind == "ai_next":
                self._ai_idx = None
            elif kind == "ai_strip_tool_calls":
                if self._ai_idx is not None:
                    role, prev = self.history[self._ai_idx]
                    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", prev, flags=re.DOTALL).strip()
                    if cleaned:
                        self.history[self._ai_idx] = (role, cleaned)
                    else:
                        self.history.pop(self._ai_idx)
                        self._ai_idx = None
                    needs_redraw = True
            elif kind == "done":
                self.thinking = False
                self._ai_idx  = None
                needs_redraw  = True
            elif kind == "error":
                self.history.append(("router", f"[Error: {event[1]}]"))
                self.thinking = False
                self._ai_idx  = None
                needs_redraw  = True
        if needs_redraw:
            self.stdscr.erase()
            self._draw_menubar()
            self._draw_chat()
            self.stdscr.refresh()

    # ------------------------------------------------------------------
    # Monitor / log background threads
    # ------------------------------------------------------------------

    def _start_monitor_thread(self):
        def _poll_system():
            while not self._stop_event.is_set():
                try:
                    with urllib.request.urlopen(MONITOR_URL, timeout=1) as resp:
                        raw = resp.read().decode()
                        try:
                            self._monitor_data = json.loads(raw)
                        except json.JSONDecodeError:
                            self._monitor_data = {"raw": raw[:200]}
                except Exception as exc:
                    self._monitor_data = {"error": str(exc)[:80]}
                self._stop_event.wait(1)

        def _poll_gpu():
            while not self._stop_event.is_set():
                try:
                    with urllib.request.urlopen(MONITOR_URL, timeout=1) as resp:
                        data = json.loads(resp.read().decode())
                        gpus = data.get("gpus", [])
                        if gpus:
                            self._gpu_history.append(gpus)
                except Exception:
                    pass
                self._stop_event.wait(0.5)

        threading.Thread(target=_poll_system, daemon=True).start()
        threading.Thread(target=_poll_gpu,   daemon=True).start()

    def _start_log_thread(self):
        def _poll():
            while self.thinking:
                try:
                    with urllib.request.urlopen(LOGS_URL, timeout=2) as resp:
                        data = json.loads(resp.read().decode())
                        entries = data.get("lines", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                        formatted = []
                        for entry in entries:
                            if isinstance(entry, dict):
                                ts   = entry.get("ts", "")
                                line = entry.get("line", "")
                                t    = ts.split("T")[-1].split(".")[0] if "T" in ts else ts
                                formatted.append(f"{t}  {line}")
                            else:
                                formatted.append(str(entry)[:200])
                        if formatted:
                            self._log_lines = formatted
                except Exception:
                    pass
                time.sleep(1)
        threading.Thread(target=_poll, daemon=True).start()

    # ------------------------------------------------------------------
    # Draw helpers
    # ------------------------------------------------------------------

    def _draw_menubar(self):
        _, w = self.stdscr.getmaxyx()
        bar = curses.color_pair(1)
        self.stdscr.attron(bar)
        try:
            self.stdscr.addstr(0, 0, " " * (w - 1))
        except curses.error:
            pass
        x = 1
        for key_name, label in MENU:
            item = f" {key_name} {label} "
            if x + len(item) < w:
                self.stdscr.addstr(0, x, item)
                x += len(item) + 1
        self.stdscr.attroff(bar)

    def _draw_home(self):
        h, w = self.stdscr.getmaxyx()
        lines = ["SakuraLang AI Terminal", "", "F1   Chat",
                 "F12  Settings", "", "ESC  Quit"]
        start_y = h // 2 - len(lines) // 2
        for i, line in enumerate(lines):
            x = max(0, (w - len(line)) // 2)
            try:
                self.stdscr.addstr(start_y + i, x, line)
            except curses.error:
                pass

    def _compute_visual_lines(self, text: str, field_w: int) -> list[tuple[int, str]]:
        """Return (char_start, display_text) for each word-wrapped visual line."""
        result = []
        paragraphs = text.split("\n")
        char_pos = 0
        for pi, para in enumerate(paragraphs):
            if not para:
                result.append((char_pos, ""))
            else:
                offset = 0
                remaining = para
                while len(remaining) > field_w:
                    split = remaining.rfind(" ", 0, field_w)
                    if split <= 0:
                        split = field_w
                    result.append((char_pos + offset, remaining[:split]))
                    skip = split + (1 if split < len(remaining) and remaining[split] == " " else 0)
                    offset += skip
                    remaining = remaining[skip:]
                result.append((char_pos + offset, remaining))
            char_pos += len(para)
            if pi < len(paragraphs) - 1:
                char_pos += 1
        return result

    def _cursor_to_visual(self, visual_lines: list, pos: int) -> tuple[int, int]:
        """Map a char-position to (visual_row, col) using precomputed visual lines."""
        for row in range(len(visual_lines) - 1):
            start_i    = visual_lines[row][0]
            start_next = visual_lines[row + 1][0]
            if start_i <= pos < start_next:
                return row, pos - start_i
        last = len(visual_lines) - 1
        return last, max(0, pos - visual_lines[last][0])

    def _draw_settings(self):
        h, w = self.stdscr.getmaxyx()
        bar  = curses.color_pair(1)

        LABEL_W = 18
        field_w = max(10, w - LABEL_W - 4)

        def _header(y: int, title: str) -> None:
            self.stdscr.attron(bar)
            try:
                hdr = f" {title} "
                self.stdscr.addstr(y, 1, hdr)
                self.stdscr.addstr(y, 1 + len(hdr), " " * (w - 2 - len(hdr)))
            except curses.error:
                pass
            self.stdscr.attroff(bar)

        y = 1
        _header(y, "Main Agent")
        y += 2
        self._draw_field(y, "  Address:       ", F_AGENT_ADDR,   LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  Working Dir:   ", F_AGENT_CWD,    LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  System Prompt: ", F_AGENT_PROMPT, LABEL_W, field_w, multiline=True)
        y += 4

        _header(y, "Classifier")
        y += 2
        self._draw_field(y, "  Address:       ", F_CLASSIFIER_ADDR,   LABEL_W, field_w, multiline=False)
        y += 2
        self._draw_field(y, "  System Prompt: ", F_CLASSIFIER_PROMPT, LABEL_W, field_w, multiline=True)
        y += 4

        footer = "  Tab: field   ↑↓ (prompt): line / (addr): field   ←→/Home/End: cursor   Enter: newline   ESC: save"
        try:
            self.stdscr.addstr(h - 1, 0, footer[:w - 1])
        except curses.error:
            pass

    def _draw_field(self, y: int, label: str, field_idx: int,
                    label_w: int, field_w: int, multiline: bool) -> None:
        active = (self.settings_focus == field_idx)
        val    = self.settings_bufs[field_idx]
        pos    = self.settings_cursors[field_idx] if active else 0

        try:
            self.stdscr.addstr(y, 1, label[:label_w])
        except curses.error:
            pass

        if not multiline:
            scroll  = max(0, pos - field_w + 1)
            view    = val[scroll:scroll + field_w]
            cur_col = pos - scroll
            self._render_line(y, 1 + label_w, view, cur_col if active else -1,
                              field_w, active)
        else:
            vlines = self._compute_visual_lines(val, field_w)
            if active:
                cur_row, cur_col_abs = self._cursor_to_visual(vlines, pos)
                v_start = max(0, cur_row - 2)
            else:
                cur_row = cur_col_abs = 0
                v_start = 0
            pad     = (0, "")
            visible = (vlines + [pad, pad])[v_start:v_start + 3]
            cont_label = " " * label_w
            for li, (_, line) in enumerate(visible):
                lbl      = label[:label_w] if li == 0 else cont_label
                abs_line = v_start + li
                in_cursor_line = active and (abs_line == cur_row)
                col = cur_col_abs if in_cursor_line else -1
                try:
                    self.stdscr.addstr(y + li, 1, lbl)
                except curses.error:
                    pass
                self._render_line(y + li, 1 + label_w, line, col,
                                  field_w, active)

    def _render_line(self, y: int, x: int, text: str, cur_col: int,
                     field_w: int, active: bool) -> None:
        """Draw one field line, painting the cursor character with A_REVERSE."""
        bg = curses.A_DIM if active else 0
        padded = text[:field_w].ljust(field_w)
        try:
            self.stdscr.addstr(y, x, padded, bg)
        except curses.error:
            pass
        if active and 0 <= cur_col <= len(text):
            ch   = text[cur_col] if cur_col < len(text) else " "
            col  = min(cur_col, field_w - 1)
            try:
                self.stdscr.addstr(y, x + col, ch, curses.A_REVERSE)
            except curses.error:
                pass

    def _draw_chat(self):
        h, w      = self.stdscr.getmaxyx()
        INPUT_H   = 3
        sep_row   = h - INPUT_H - 1
        input_top = h - INPUT_H
        chat_top  = 1
        chat_h    = sep_row - chat_top

        # Right quarter = monitor panel; left = chat
        show_panel = (w >= 60)
        if show_panel:
            panel_w  = max(20, w // 4)
            vsep_col = w - panel_w - 1
            chat_x   = 0
            chat_w   = max(1, vsep_col - 1)
        else:
            vsep_col = 0
            chat_x   = 0
            chat_w   = w - 2

        field_w = w - 3  # input always spans full width

        if show_panel:
            self._draw_monitor_panel(chat_top, sep_row, panel_w, panel_x=vsep_col + 1)
            for row in range(chat_top, sep_row):
                try:
                    self.stdscr.addch(row, vsep_col, ord('|'), curses.color_pair(4))
                except curses.error:
                    pass

        # Chat history
        lines = []
        for role, content in self.history:
            pair   = curses.color_pair(ROLE_PAIR.get(role, 0))
            prefix = ROLE_PREFIX.get(role, "")
            for wrapped in self._wrap(prefix + content, chat_w):
                lines.append((pair, wrapped))

        total      = len(lines)
        max_scroll = max(0, total - chat_h)
        self.chat_scroll = min(self.chat_scroll, max_scroll)
        end     = total - self.chat_scroll
        start   = max(0, end - chat_h)
        visible = lines[start:end]

        # Horizontal separator (full width)
        self.stdscr.attron(curses.color_pair(1))
        try:
            note = f" ↑ {self.chat_scroll} " if self.chat_scroll > 0 else ""
            sep  = ("-" * (w - 1 - len(note))) + note if note else "-" * (w - 1)
            self.stdscr.addstr(sep_row, 0, sep[:w - 1])
        except curses.error:
            pass
        self.stdscr.attroff(curses.color_pair(1))

        for i, (pair, line) in enumerate(visible):
            y = chat_top + i
            if y < sep_row:
                try:
                    self.stdscr.addstr(y, chat_x, line, pair)
                except curses.error:
                    pass

        if self.thinking:
            display = self._log_lines[-INPUT_H:] if self._log_lines else ["thinking..."]
            for li, log_text in enumerate(display):
                try:
                    self.stdscr.addstr(input_top + li, 0, f"  {log_text.strip()}"[:w - 1])
                except curses.error:
                    pass
            return

        vlines           = self._compute_visual_lines(self.input_buf, field_w)
        cur_row, cur_col = self._cursor_to_visual(vlines, self.input_cursor)
        v_start          = max(0, cur_row - (INPUT_H - 1))
        pad              = (0, "")
        visible_in       = (vlines + [pad, pad])[v_start:v_start + INPUT_H]

        for li, (_, line) in enumerate(visible_in):
            row_y  = input_top + li
            prefix = "> " if li == 0 else "  "
            try:
                self.stdscr.addstr(row_y, 0, prefix)
            except curses.error:
                pass
            abs_vline = v_start + li
            col = cur_col if (abs_vline == cur_row) else -1
            self._render_line(row_y, 2, line, col, field_w, True)

        vis_li = cur_row - v_start
        if 0 <= vis_li < INPUT_H:
            self.stdscr.move(input_top + vis_li, min(2 + cur_col, w - 1))

    def _draw_monitor_panel(self, top: int, bottom: int, panel_w: int, panel_x: int = 1) -> None:
        pair    = curses.color_pair(4)
        avail_w = panel_w - 2

        title = "[ Monitor ]"
        try:
            self.stdscr.addstr(top, panel_x, title[:avail_w], pair | curses.A_BOLD)
        except curses.error:
            pass

        data = self._monitor_data
        row  = top + 1

        if not data:
            try:
                self.stdscr.addstr(row, panel_x, "connecting..."[:avail_w], pair)
            except curses.error:
                pass
            return

        for line in self._format_monitor_lines(data, avail_w):
            if row >= bottom:
                break
            try:
                self.stdscr.addstr(row, panel_x, line, pair)
            except curses.error:
                pass
            row += 1

    def _avg_gpu(self, gpu_idx: int, key: str) -> float | None:
        samples = [
            snap[gpu_idx].get(key)
            for snap in self._gpu_history
            if gpu_idx < len(snap) and snap[gpu_idx].get(key) is not None
        ]
        return sum(samples) / len(samples) if samples else None

    def _format_monitor_lines(self, data: dict | list | str, width: int) -> list[str]:
        if not isinstance(data, dict):
            if isinstance(data, list):
                return [str(x)[:width] for x in data]
            return [p[:width] for p in str(data).split("\n")]

        lines = []

        # Timestamp
        ts = data.get("updated_at", "")
        if ts:
            time_part = str(ts).split("T")[-1] if "T" in str(ts) else str(ts)
            lines.append(f"@ {time_part}"[:width])

        # System
        sys = data.get("system", {})
        if sys:
            lines.append("- system"[:width])
            cpu = sys.get("cpu_percent")
            if cpu is not None:
                lines.append(f"  CPU   {cpu:.1f}%"[:width])
            ram_u = sys.get("ram_used_gib")
            ram_t = sys.get("ram_total_gib")
            ram_p = sys.get("ram_percent")
            if ram_u is not None and ram_t is not None:
                lines.append(f"  RAM   {ram_u:.1f}/{ram_t:.1f} GiB"[:width])
            if ram_p is not None:
                lines.append(f"        {ram_p:.1f}%"[:width])

        # GPUs — VRAM from latest snapshot, loads averaged over last 10s
        gpus = data.get("gpus", [])
        if not gpus and self._gpu_history:
            gpus = self._gpu_history[-1]
        if gpus:
            lines.append("- gpu"[:width])
            for i, gpu in enumerate(gpus):
                name = gpu.get("name", f"GPU {i}")
                short = (name
                         .replace("AMD Radeon ", "")
                         .replace("NVIDIA GeForce ", "")
                         .replace("NVIDIA ", ""))
                lines.append(f"  [{i}] {short}"[:width])

                vram_u = gpu.get("vram_used_mib")
                vram_t = gpu.get("vram_total_mib")
                vram_p = gpu.get("vram_percent")
                if vram_u is not None and vram_t is not None:
                    lines.append(f"  VRAM  {vram_u}/{vram_t} MiB"[:width])
                if vram_p is not None:
                    lines.append(f"        {vram_p:.1f}%"[:width])

                util  = self._avg_gpu(i, "util_percent")
                power = self._avg_gpu(i, "power_watts")
                temp  = self._avg_gpu(i, "temperature_c")
                clock = self._avg_gpu(i, "core_clock_mhz")
                parts = []
                if util  is not None: parts.append(f"{util:.0f}%")
                if power is not None: parts.append(f"{power:.0f}W")
                if temp  is not None: parts.append(f"{temp:.0f}°C")
                if parts:
                    lines.append(f"  {' '.join(parts)}"[:width])
                if clock is not None:
                    lines.append(f"  {clock:.0f} MHz"[:width])

        return lines

    @staticmethod
    def _wrap(text, width):
        result = []
        for paragraph in text.split("\n"):
            if not paragraph:
                result.append("")
                continue
            while len(paragraph) > width:
                split = paragraph.rfind(" ", 0, width)
                if split <= 0:
                    split = width
                result.append(paragraph[:split])
                paragraph = "  " + paragraph[split:].lstrip()
            result.append(paragraph)
        return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    with SqliteSaver.from_conn_string("chat_history.db") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        def _run(stdscr):
            App(stdscr, graph).run()

        curses.wrapper(_run)


if __name__ == "__main__":
    main()
