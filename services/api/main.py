from fastapi import FastAPI
from api.routers.elms import router as elm_router

app = FastAPI()
app.include_router(elm_router)