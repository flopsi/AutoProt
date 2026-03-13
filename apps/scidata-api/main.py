from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.analysis_router import router as analysis_router

app = FastAPI(title="Proteomics QC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router, prefix="/api")

@app.get("/")
def read_root():
    return {"status": "ok"}
