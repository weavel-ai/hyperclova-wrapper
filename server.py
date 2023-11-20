from enum import Enum
import os
from time import time
from typing import Any, Dict, List
from uuid import uuid4
import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from constants import CLOVA_STUDIO_API_ENDPOINT, HYPERCLOVA_MODEL_NAME
from rich import print

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()


class Role(str, Enum):
    system: str = "system"
    assistant: str = "assistant"
    user: str = "user"


class Message(BaseModel):
    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str = HYPERCLOVA_MODEL_NAME


@app.get("/health")
def health():
    """Health check endpoint."""
    return Response(status_code=200)


@app.post("/{project_name}/chat/completions")
async def chat_completion(
    body: ChatCompletionRequest,
    project_name: str,
):
    url = f"{CLOVA_STUDIO_API_ENDPOINT}/{project_name}/v1/chat-completions/{body.model}"
    headers = {
        "X-NCP-CLOVASTUDIO-API-KEY": os.environ.get("NCP_CLOVASTUDIO_API_KEY"),
        "X-NCP-APIGW-API-KEY": os.environ.get("NCP_APIGW_API_KEY"),
        "Content-Type": "application/json",
    }
    req_data = {
        "messages": [message.model_dump() for message in body.messages],
        "includeAIFilters": False,
    }
    try:
        async with httpx.AsyncClient() as _client:
            response = await _client.request(
                "POST",
                url,
                headers=headers,
                data=req_data,
            )
        if not response:
            print(f"[red]Error: {response}[/red]")
        if response.status_code == 200:
            res_data: Dict[str, Any] = response.json()
            result: Dict[str, Any] = res_data["result"]
        else:
            return HTTPException(
                status_code=response.status_code, detail=response.json()
            )
    except Exception as error:
        print(f"[red]Error: {error}[/red]")
        return HTTPException(status_code=500, detail=str(error))

    return JSONResponse(
        content={
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": result["stopReason"],
                    "index": 0,
                    "message": result["message"],
                }
            ],
            "id": f"chatcmpl-{uuid4()}",  # Unique identifier for completion (placeholder, not used)
            "created": int(time()),  # Unix timestamp in seconds
            "model": body.model,
            "usage": {
                "completion_tokens": result["outputLength"],
                "prompt_tokens": result["inputLength"],
                "total_tokens": result["inputLength"] + result["outputLength"],
            },
        }
    )
