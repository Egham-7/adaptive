from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.model_selection import router as model_selector_router
from api.routes.classifier import router as classifier_router

# Create the FastAPI app
app = FastAPI(
    title="AI Model Selection API",
    description="API for selecting and configuring AI models based on prompt analysis",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with prefixes
app.include_router(model_selector_router, prefix="/api/model", tags=["Model Selection"])
app.include_router(classifier_router, prefix="/api/classify", tags=["Classification"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Adaptive",
        "docs": "/docs",
        "redoc": "/redoc",
    }
