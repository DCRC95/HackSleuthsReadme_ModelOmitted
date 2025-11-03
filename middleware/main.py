import os
import orjson
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, ORJSONResponse, FileResponse
from starlette.middleware.gzip import GZipMiddleware

load_dotenv()

app = FastAPI(default_response_class=ORJSONResponse)

app.add_middleware(GZipMiddleware, minimum_size=1_000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
client = MongoClient(mongo_uri)
db = client["mlresults"]
collection = db["experiments"]


@app.get("/hack_analysis/main")
async def generate_analysis_data():
    # Fetch the "from" field from
    docs = collection.find_one({"source_file": "hack_analysis_hashes.json"})
    docs = docs.get("content", {})
    flat = []
    for outer in docs:
        row = {"id": outer}  # keep the label if you need it
        row.update(docs[outer])  # add all inner fields
        flat.append(row)
    return flat


@app.get("/hack_analysis/sars-list")
async def generate_sars_data():
    docs = collection.find_one(
        {"source_file": "output_crypto_transactions_enhanced.json"}
    )
    docs = docs.get("content", [])

    async def stream():
        yield b"["
        first = True
        for doc in docs:
            if not first:
                yield b","
            else:
                first = False
            yield orjson.dumps(doc)  # already bytes
        yield b"]"

    return StreamingResponse(stream(), media_type="application/json")


@app.get("/hack_analysis/sars-list/{hack_name}")
async def generate_sars_data(hack_name: str):
    docs = collection.aggregate(
        [
            {"$match": {"source_file": "output_crypto_transactions_enhanced.json"}},
            {"$unwind": "$content"},
            {"$match": {"content.hack_name": hack_name}},
        ]
    )

    async def stream():
        yield b"["
        first = True
        for doc in docs:
            if not first:
                yield b","
            else:
                first = False
            yield orjson.dumps(doc.get("content", {}))  # already bytes
        yield b"]"

    return StreamingResponse(stream(), media_type="application/json")


@app.get("/hack_analysis/network-visualization/{hack_name}")
async def fetch_visualization_image(hack_name: str):
    image_file_name = f'{hack_name.replace(" ", "_")}_network.png'
    pipeline = [
        {"$match": {"original_file": "network_visualizations.json"}},
        {"$unwind": "$content"},
        {"$project": {"kvPairs": {"$objectToArray": "$content"}}},
        {"$unwind": "$kvPairs"},
        {"$match": {"kvPairs.k": image_file_name}},
        {"$project": {"content": "$kvPairs.v", "_id": 0}},
    ]
    docs = list(collection.aggregate(pipeline))
    docs = docs[0]
    return docs.get("content", {})


@app.get("/hack_analysis/offshore-analysis-summary/{hack_name}")
async def fetch_explanation(hack_name: str):
    docs = collection.find_one(
        {"hack_name": hack_name, "document_type": "explanation_text"}
    )
    return docs.get("content", {})


@app.get("/hack_analysis/offshore-analysis-visualization/{hack_name}")
async def fetch_image(hack_name: str):
    image_file_name = f'network_diagram_{hack_name.replace(" ", "_")}'
    docs = collection.find_one(
        {"source_file": image_file_name, "document_type": "network_diagram"}
    )
    return docs.get("content", {})


@app.get("/report/{name}")
def get_report(name: str):
    REPORTS_DIR = "/reports"
    path = os.path.join(REPORTS_DIR, f"{name} Report.docx")
    if not os.path.exists(path):
        return {"error": "Report not found"}
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"{name} Report.docx",
    )
