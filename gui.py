import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

import requests

from rag import answer, OLLAMA_MODEL

OLLAMA_STATUS_URL = "http://localhost:11434/api/tags"


def _check_ollama():
    try:
        r = requests.get(OLLAMA_STATUS_URL, timeout=3)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return True, models
    except Exception:
        return False, []

K = 5


def _run_query(question, model, callback):
    try:
        result = answer(question, k=K, model=model)
        callback(result, None)
    except Exception as exc:
        callback(None, str(exc))


class DocuBotApp:
    def __init__(self, root):
        self.root = root
        root.title("DocuBot")
        root.minsize(640, 520)

        self.display = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, state=tk.DISABLED,
            font=("Helvetica", 11), padx=10, pady=10,
        )
        self.display.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 0))

        self.display.tag_config("question", font=("Helvetica", 11, "bold"), foreground="#1a73e8")
        self.display.tag_config("answer",   font=("Helvetica", 11))
        self.display.tag_config("sources",  font=("Helvetica", 10), foreground="#757575")
        self.display.tag_config("thinking", font=("Helvetica", 11, "italic"), foreground="#aaaaaa")
        self.display.tag_config("error",    font=("Helvetica", 11), foreground="#d32f2f")

        input_frame = tk.Frame(root)
        input_frame.pack(fill=tk.X, padx=12, pady=12)

        self.entry = tk.Entry(input_frame, font=("Helvetica", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        self.entry.bind("<Return>", self._on_submit)
        self.entry.focus()

        self.btn = tk.Button(
            input_frame, text="Ask", command=self._on_submit,
            font=("Helvetica", 11), padx=14,
        )
        self.btn.pack(side=tk.LEFT, padx=(8, 0))

        status_bar = tk.Frame(root, bd=1, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, padx=0, pady=(0, 0))

        self.status_dot = tk.Label(status_bar, text="●", font=("Helvetica", 10), fg="#aaaaaa")
        self.status_dot.pack(side=tk.LEFT, padx=(8, 2), pady=3)

        self.status_label = tk.Label(status_bar, text="Ollama: checking…", font=("Helvetica", 9), fg="#757575")
        self.status_label.pack(side=tk.LEFT, pady=3)

        self.selected_model = tk.StringVar(value=OLLAMA_MODEL)
        self.model_dropdown = ttk.Combobox(
            status_bar, textvariable=self.selected_model,
            state="readonly", font=("Helvetica", 9), width=18,
        )
        self.model_dropdown.pack(side=tk.RIGHT, padx=(0, 8), pady=2)

        tk.Label(status_bar, text="Model:", font=("Helvetica", 9), fg="#757575").pack(side=tk.RIGHT, pady=2)

        self._refresh_status()

    def _refresh_status(self):
        threading.Thread(target=self._check_and_update_status, daemon=True).start()

    def _check_and_update_status(self):
        online, models = _check_ollama()
        self.root.after(0, self._update_status_ui, online, models)

    def _update_status_ui(self, online, models):
        if online:
            self.status_dot.configure(fg="#2e7d32")
            self.status_label.configure(text="Ollama: Online")
            if models:
                self.model_dropdown.configure(values=models)
                if self.selected_model.get() not in models:
                    self.selected_model.set(models[0])
        else:
            self.status_dot.configure(fg="#d32f2f")
            self.status_label.configure(text="Ollama: Offline")
            self.model_dropdown.configure(values=[])

    # ------------------------------------------------------------------
    def _append(self, text, tag=None):
        self.display.configure(state=tk.NORMAL)
        self.display.insert(tk.END, text, tag or "")
        self.display.configure(state=tk.DISABLED)
        self.display.see(tk.END)

    def _on_submit(self, _event=None):
        question = self.entry.get().strip()
        if not question:
            return

        self.entry.delete(0, tk.END)
        self.entry.configure(state=tk.DISABLED)
        self.btn.configure(state=tk.DISABLED)

        self._append(f"Q: {question}\n", "question")

        # Mark position so we can remove the "Thinking…" line once done
        self.display.configure(state=tk.NORMAL)
        self.display.mark_set("thinking_start", "end-1c")
        self.display.configure(state=tk.DISABLED)

        self._append("Thinking…\n", "thinking")

        self._refresh_status()

        threading.Thread(
            target=_run_query, args=(question, self.selected_model.get(), self._on_result), daemon=True
        ).start()

    def _on_result(self, result, error):
        self.root.after(0, self._render_result, result, error)

    def _render_result(self, result, error):
        self.display.configure(state=tk.NORMAL)
        self.display.delete("thinking_start", tk.END)
        self.display.configure(state=tk.DISABLED)

        if error:
            self._append(f"Error: {error}\n\n", "error")
        else:
            self._append(f"A: {result['answer']}\n", "answer")
            self._append(f"Sources: {', '.join(result['sources'])}\n\n", "sources")

        self.entry.configure(state=tk.NORMAL)
        self.btn.configure(state=tk.NORMAL)
        self.entry.focus()


if __name__ == "__main__":
    root = tk.Tk()
    DocuBotApp(root)
    root.mainloop()
