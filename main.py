import os
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from factory_core import ModelFactoryEngine

app = FastAPI(title="NeuralForge Backend Engine", version="2.1")

API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
NEURALFORGE_MASTER_KEY = os.environ.get("NEURALFORGE_MASTER_KEY", "neuralforge-secret-secure-key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key == NEURALFORGE_MASTER_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Unauthorized: Invalid master API key.")

project_ledger = {}

class BuildRequest(BaseModel):
    project_id: str
    user_prompt: str
    fetch_web: bool = False
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def background_build_pipeline(project_id: str, user_prompt: str, fetch_web: bool, base_model: str):
    try:
        project_ledger[project_id] = {"status": "Building", "timestamp": time.time()}
        factory = ModelFactoryEngine(output_dir=f"./temp_{project_id}")
        web_data = factory.fetch_web_knowledge(user_prompt) if fetch_web else ""
        dataset = factory.build_dataset(user_prompt, web_data)
        factory.train_lora_adapter(base_model, dataset)

        project_ledger[project_id] = {
            "status": "Ready",
            "timestamp": time.time(),
            "adapter_path": f"./temp_{project_id}/adapter_model"
        }
    except Exception as e:
        project_ledger[project_id] = {"status": f"Failed: {str(e)}", "timestamp": time.time()}

@app.post("/api/factory/build")
def trigger_build(req: BuildRequest, background_tasks: BackgroundTasks, api_key: str = Security(verify_api_key)):
    if req.project_id in project_ledger and project_ledger[req.project_id]["status"] == "Building":
        raise HTTPException(status_code=400, detail="A build pipeline for this project ID is already active.")

    background_tasks.add_task(background_build_pipeline, req.project_id, req.user_prompt, req.fetch_web, req.base_model)
    return {"status": "Success", "message": "Background build job queued.", "project_id": req.project_id}

@app.get("/api/factory/status/{project_id}")
def check_status(project_id: str, api_key: str = Security(verify_api_key)):
    if project_id not in project_ledger:
        raise HTTPException(status_code=404, detail="Project ID registry entry not found.")
    return project_ledger[project_id]
