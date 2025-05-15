from fastapi import APIRouter, Request
from services.api.schemas.models import Model

router = APIRouter(prefix="/projects/{project_id}/models")

@router.get("")
async def get_models(project_id: str):
    # Return details about models being used by this project
    # Could be eg the ID, type of model, the accuracy, the version. link to mlflow / simvue instance, etc...
    pass

@router.post("")
async def get_models(project_id: str, params: Model):
    # Create a new model for this project
    # Creates (but doesnt train?) a new model and stores it somewhere
    # Returns the model ID for use later
    pass

@router.get("/{model_id}")
async def get_model(project_id: str, model_id: str):
    # Return details about this specific model
    # Could be eg the type of model, the accuracy, the version. link to mlflow / simvue instance, etc...
    pass

@router.delete("/{model_id}")
async def delete_model(project_id: str, model_id: str, version: int = None):
    # Delete this model
    # If a specific version number is specified, just delete that version, else delete the whole thing
    pass

@router.get("/{model_id}/train")
async def get_training_info(project_id: str, model_id: str):
    # Get current status of model training
    pass

@router.put("/{model_id}/train")
async def train_model(project_id: str, model_id: str):
    # Start training of model
    # This triggers the ModelRunner, which goes and gets all validated samples from DB
    # Splits these into training / validation sets, passes to ModelWorker
    # ModelWorker then has an instance of a DataLoader which it uses to collect data for above samples
    # Retrains, tracked by mlflow / simvue
    # Once finished, runs inference over some set of samples and stores unvalidated results into DB
    # NON BLOCKING ENDPOINT, so does not wait for training to complete
    pass

@router.delete("/{model_id}/train")
async def stop_model_training(project_id: str, model_id: str):
    # Stop training of this model
    pass