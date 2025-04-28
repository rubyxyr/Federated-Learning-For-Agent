from server.server import BaseServer
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
import uvicorn
import os

app = FastAPI()


@app.post("/update_wight")
async def update(request: Request):
    server = BaseServer("config.yaml")
    server.update()


@app.get("/latest_weight")
async def update(request: Request):
    server = BaseServer("config.yaml")
    filename = f'{server.latest_version}-adapter_model.bin'
    return FileResponse(os.path.join(server.output, server.latest_version, 'adapter_model.bin'),
                        media_type='application/octet-stream',
                        filename=filename,
                        headers={"Content-Disposition": f"attachment; filename={filename}"})


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)
