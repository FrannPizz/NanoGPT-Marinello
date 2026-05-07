#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Main: loads checkpoint and launches Gradio UI.

from checkpoint import load_checkpoint
from ui import demo

if __name__ == "__main__":
    print("🛥️ Starting NanoGPT Marinello...")
    print(load_checkpoint("nanoGPT_checkpoint.pth"))
    print("🌐 Opening web interface...")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        debug=True,
        show_error=True,
        quiet=False,
        inbrowser=True
    )
