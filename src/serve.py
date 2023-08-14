# a fastapi wrapper
import yaml
import uvicorn
from fastapi import FastAPI
from chatbot import Chatbot
from pydantic import BaseModel


class Sentence(BaseModel):
    text: str


with open('config.yaml') as inp:
    config = yaml.load(inp, Loader=yaml.SafeLoader)
model = Chatbot(config)
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Running!"}


@app.get("/start")
async def start():
    return model.on_start()


@app.get("/metrics")
async def get_metrics():
    return model.get_metrics()

@app.get("/exit")
async def on_exit():
    model.restart()


@app.post("/post")
async def post(data: Sentence):
    # TODO generate response
    return model.respond(data.text)


#uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=config['server_port'])