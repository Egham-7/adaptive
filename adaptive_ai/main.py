from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator

# Import routers
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


# Initialize Prometheus metrics
@app.on_event("startup")
async def startup():
    # Set up Prometheus instrumentator
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[".*admin.*", "/metrics"],
        env_var_name="ENABLE_METRICS",
    )

    # Add custom metrics
    instrumentator.add(metrics_handlers=True)

    # Expose the metrics
    instrumentator.instrument(app).expose(app, include_in_schema=True, should_gzip=True)


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Adaptive",
        "docs": "/docs",
        "redoc": "/redoc",
    }


# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
