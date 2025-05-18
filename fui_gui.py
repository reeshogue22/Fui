import tkinter as tk
from tkinter.scrolledtext import ScrolledText

from fui import Fui


class FuiGUI:
    """Simple Tkinter based GUI for interacting with Fui."""

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("Fui Chat")
        self.fui = Fui()
        self._build_widgets()

    def _build_widgets(self) -> None:
        self.display = ScrolledText(self.master, wrap=tk.WORD, height=20, width=60, state=tk.DISABLED)
        self.display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(self.master, textvariable=self.entry_var)
        self.entry.bind("<Return>", lambda event: self._on_send())
        self.entry.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)

        self.send_button = tk.Button(self.master, text="Send", command=self._on_send)
        self.send_button.pack(side=tk.LEFT, padx=10, pady=10)

    def _on_send(self) -> None:
        message = self.entry_var.get().strip()
        self.entry_var.set("")
        if not message:
            return

        self._append_message("User", message)
        self.fui.add_message("user", message)

        structured = self.fui.structured_prompt(self.fui.chat_history, role="fui")
        response = self.fui.get_response(structured)
        self.fui.add_message("fui", response)
        self._append_message("Fui", response)

    def _append_message(self, role: str, content: str) -> None:
        self.display.configure(state=tk.NORMAL)
        self.display.insert(tk.END, f"{role}: {content}\n")
        self.display.configure(state=tk.DISABLED)
        self.display.see(tk.END)


def main() -> None:
    root = tk.Tk()
    FuiGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
