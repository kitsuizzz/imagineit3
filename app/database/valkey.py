import os
import valkey
from fastapi import FastAPI


def init_valkey(app: FastAPI):
    client = valkey.Valkey(
        host=os.getenv("VALKEY_HOST", "localhost"),
        port=int(os.getenv("VALKEY_PORT", "6379")),
        password=os.getenv("VALKEY_PASSWORD", ""),
        decode_responses=True,
    )
    if client.ping():
        print("Valkey is up")
        app.state.valkey = client
