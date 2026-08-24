import uvicorn
import webbrowser
import os
from threading import Timer

def open_browser():
    webbrowser.open_new("http://localhost:8000")

if __name__ == "__main__":
    print("=" * 70)
    print("  Starting SOSA-Aligned Radar Emitter Recognition Platform")
    print("  Open-Set Radar EW System | VS Code Launch Environment")
    print("=" * 70)
    
    Timer(1.5, open_browser).start()
    
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
