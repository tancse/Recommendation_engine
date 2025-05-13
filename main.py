from fastapi import FastAPI, Query
from pydantic import BaseModel
from model import rank_products
from typing import Union

app = FastAPI()

class QueryInput(BaseModel):
    keyword: str | None = None

@app.get("/recommend/{query}")
def read_item(query: str, offset : int = Query(0, alias="page", ge=0), top_k : int = Query(10 , le = 50), q: Union[str, None] = None):
    results = rank_products(query, offset, top_k)
    return {"results" : results, "offset" : offset, "to_k" : top_k}
