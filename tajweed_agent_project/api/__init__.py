from fastapi import FastAPI

app = FastAPI(title="Tajweed Agent API")

@app.get("/health")
def health():
    return {"ok": True}
