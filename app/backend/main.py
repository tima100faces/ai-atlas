"""
AI Atlas — FastAPI Backend

File-first learning platform. Serves Markdown content, progress YAML files,
and static frontend assets. Simple password auth.
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import yaml
import os

# --- Config ---
PASSWORD = os.getenv("AI_ATLAS_PASSWORD", "timohin2026")
CONTENT_DIR = Path(os.getenv("CONTENT_DIR", str(Path(__file__).resolve().parent.parent.parent / "content")))
PROGRESS_DIR = Path(os.getenv("PROGRESS_DIR", str(Path(__file__).resolve().parent.parent.parent / "progress")))

app = FastAPI(title="AI Atlas", version="0.1.0")

# CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth ---
def check_auth(request: Request):
    """Simple password check via cookie or header."""
    cookie = request.cookies.get("ai_atlas_auth")
    header = request.headers.get("X-AI-Atlas-Auth")
    if cookie == PASSWORD or header == PASSWORD:
        return True
    raise HTTPException(status_code=401, detail="Unauthorized")


# --- Health ---
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


# --- Auth endpoints ---
@app.post("/api/auth/login")
async def login(request: Request):
    body = await request.json()
    password = body.get("password", "")
    if password != PASSWORD:
        raise HTTPException(status_code=401, detail="Wrong password")
    response = JSONResponse({"status": "ok"})
    response.set_cookie(key="ai_atlas_auth", value=PASSWORD, httponly=True, samesite="lax")
    return response


@app.post("/api/auth/logout")
def logout():
    response = JSONResponse({"status": "ok"})
    response.delete_cookie("ai_atlas_auth")
    return response


@app.get("/api/auth/check")
def auth_check(_: None = Depends(check_auth)):
    return {"authenticated": True}


# --- Content endpoints ---
@app.get("/api/content/book/{language}/{chapter_id}")
def get_chapter(language: str, chapter_id: str, _: None = Depends(check_auth)):
    """Serve a book chapter as raw markdown."""
    chapter_path = CONTENT_DIR / "book" / language / f"{chapter_id}.md"
    if not chapter_path.exists():
        raise HTTPException(status_code=404, detail="Chapter not found")
    return {"content": chapter_path.read_text(encoding="utf-8")}


@app.get("/api/content/book/meta")
def get_book_meta(_: None = Depends(check_auth)):
    """Return book metadata (ToC, chapter list)."""
    meta_path = CONTENT_DIR / "book" / "meta.yaml"
    if not meta_path.exists():
        return {"chapters": []}
    with open(meta_path) as f:
        return yaml.safe_load(f)


# --- Progress endpoints ---
def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: dict):
    with open(path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


@app.get("/api/progress/book")
def get_book_progress(_: None = Depends(check_auth)):
    return read_yaml(PROGRESS_DIR / "book.yaml")


@app.put("/api/progress/book")
async def update_book_progress(request: Request, _: None = Depends(check_auth)):
    data = await request.json()
    write_yaml(PROGRESS_DIR / "book.yaml", data)
    return {"status": "ok"}


@app.get("/api/progress/courses")
def get_courses_progress(_: None = Depends(check_auth)):
    return read_yaml(PROGRESS_DIR / "courses.yaml")


@app.put("/api/progress/courses")
async def update_courses_progress(request: Request, _: None = Depends(check_auth)):
    data = await request.json()
    write_yaml(PROGRESS_DIR / "courses.yaml", data)
    return {"status": "ok"}


@app.get("/api/progress/overall")
def get_overall_progress(_: None = Depends(check_auth)):
    progress = read_yaml(PROGRESS_DIR / "overall.yaml")
    if not progress:
        # Compute from book progress
        book = read_yaml(PROGRESS_DIR / "book.yaml")
        chapters = book.get("chapters", {})
        total = len(chapters)
        done = sum(1 for c in chapters.values() if c.get("status") == "done")
        progress = {
            "book_percent": round(done / total * 100) if total > 0 else 0,
            "current_chapter_id": book.get("overall", {}).get("current_chapter_id"),
            "current_language": book.get("overall", {}).get("current_language", "ru"),
        }
    return progress


# --- Static files (will serve React build in production) ---
STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
