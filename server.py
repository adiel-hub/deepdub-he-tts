from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import StreamingResponse, Response
import os
from dotenv import load_dotenv
import uvicorn
import subprocess
import json
import asyncio
import time
import base64
import binascii
from typing import Optional
import websockets
from pathlib import Path
import httpx


env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

deepdub_api_key = os.getenv("DEEPDUB_API_KEY")
vapi_api_key = os.getenv("VAPI_API_KEY")
DEEPDUB_WS_URL = "wss://wsapi.deepdub.ai/open"
VAPI_BASE_URL = "https://api.vapi.ai"

PORT = int(os.getenv("PORT", 4000))  

app = FastAPI()

print(f"🚀 Server Starting on Port {PORT}...")

def ffmpeg_wav_or_mp3_to_pcm16k(blob: bytes) -> bytes:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        "pipe:1",
    ]
    p = subprocess.run(cmd, input=blob, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0 or not p.stdout:
        err = p.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed: {err[:400]}")
    return p.stdout

def looks_like_wav(b: bytes) -> bool:
    return len(b) >= 12 and b[0:4] == b"RIFF" and b[8:12] == b"WAVE"

@app.post("/to-speech")
async def to_speech(request: Request):
    t0 = time.perf_counter()
    payload = await request.json()
    msg = payload.get("message") or {}

    text = payload.get("text") or msg.get("text") or msg.get("content")
    if not text:
        return Response(status_code=200)

    q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=400)

    async def producer():
        ws_chunks = 0
        total_decoded = 0
        total_pcm = 0
        first = True

        try:
            async with websockets.connect(
                DEEPDUB_WS_URL,
                additional_headers={"x-api-key": deepdub_api_key},
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            ) as ws:
                req = {
                    "action": "text-to-speech",
                    "locale": "he-IL",
                    "voicePromptId": "cd91dbf2-7265-420b-b8fd-9b90f2555d02_prompt-reading-neutral",
                    "model": "dd-etts-3.0-preview",
                    "targetText": text,
                    "cleanAudio": True,
                    "realTime": True
                }

                print(f"\n📨 /to-speech | text_len={len(text)}")
                await ws.send(json.dumps(req))

                while True:
                    raw_msg = await ws.recv()
                    msgj = json.loads(raw_msg)

                    ws_chunks += 1
                    idx = msgj.get("index")
                    gid = msgj.get("generationId")
                    is_finished = bool(msgj.get("isFinished"))
                    b64 = msgj.get("data")

                    print(f"📦 WS chunk | gid={gid} idx={idx} finished={is_finished} has_data={bool(b64)}")

                    if b64:
                        audio_bytes = base64.b64decode(b64)
                        total_decoded += len(audio_bytes)

                        if first:
                            ttfa = (time.perf_counter() - t0) * 1000
                            print(f"⚡ TTFA={ttfa:.1f}ms | first_size={len(audio_bytes)} bytes")
                            print("🔎 ascii4:", audio_bytes[:4])
                            print("🔎 first16(hex):", binascii.hexlify(audio_bytes[:16]).decode())
                            first = False

                        if looks_like_wav(audio_bytes):
                            pcm = await asyncio.to_thread(ffmpeg_wav_or_mp3_to_pcm16k, audio_bytes)
                            total_pcm += len(pcm)
                            await q.put(pcm)
                        else:
                            print("⚠️ chunk not WAV; skipping (or handle separately)")

                    if is_finished:
                        break

            t_total = (time.perf_counter() - t0) * 1000
            print(f"🏁 WS done | chunks={ws_chunks} decoded={total_decoded} pcm={total_pcm} total_time={t_total:.1f}ms")

        except Exception as e:
            print(f"❌ producer error: {e}")
        finally:
            await q.put(None)

    asyncio.create_task(producer())

    async def stream_pcm():
        sent = 0
        while True:
            chunk = await q.get()
            if chunk is None:
                print(f"🏁 stream done | total_sent_pcm={sent}")
                break
            sent += len(chunk)
            yield chunk

    return StreamingResponse(
        stream_pcm(),
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/assistant/{assistant_id}/enable")
async def enable_hebrew_tts(
    assistant_id: str,
    request: Request,
    vapi_key: Optional[str] = Query(
        None,
        description="VAPI API key. If not provided, uses server's VAPI_API_KEY env variable."
    ),
    server_url: Optional[str] = Query(
        None,
        description="Custom TTS server URL. If not provided, auto-detects from request."
    ),
):
    """Enable Hebrew TTS for a VAPI assistant."""

    # Use provided API key or fall back to server config
    api_key = vapi_key or vapi_api_key

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="VAPI API key required. Provide 'vapi_key' query parameter or configure VAPI_API_KEY on server."
        )

    # Auto-detect server URL from request if not provided
    if not server_url:
        # Get the base URL from the incoming request
        server_url = str(request.base_url).rstrip('/')

    # Build TTS endpoint URL
    tts_url = f"{server_url}/to-speech"

    # Voice configuration for custom Deepdub TTS
    voice_config = {
        "voice": {
            "provider": "custom-voice",
            "server": {
                "url": tts_url,
                "timeoutSeconds": 30,
            },
            "inputMinCharacters": 1,
            "inputPunctuationBoundaries": [".", "!", "?", ":", ";"],
        }
    }

    print(f"📝 Updating assistant {assistant_id} with Hebrew TTS: {tts_url}")

    async with httpx.AsyncClient() as client:
        try:
            # First, get current assistant to verify it exists
            get_response = await client.get(
                f"{VAPI_BASE_URL}/assistant/{assistant_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )

            if get_response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Assistant {assistant_id} not found")
            elif get_response.status_code != 200:
                raise HTTPException(
                    status_code=get_response.status_code,
                    detail=f"Failed to get assistant: {get_response.text}"
                )

            current_assistant = get_response.json()
            assistant_name = current_assistant.get("name", "Unknown")

            # Patch the assistant with new voice config
            patch_response = await client.patch(
                f"{VAPI_BASE_URL}/assistant/{assistant_id}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=voice_config,
                timeout=30.0,
            )

            if patch_response.status_code != 200:
                raise HTTPException(
                    status_code=patch_response.status_code,
                    detail=f"Failed to update assistant: {patch_response.text}"
                )

            print(f"✅ Assistant '{assistant_name}' updated successfully")

            return {
                "success": True,
                "assistant_id": assistant_id,
                "assistant_name": assistant_name,
                "message": f"Assistant '{assistant_name}' updated to use Deepdub Hebrew TTS at {tts_url}"
            }

        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Request to VAPI timed out")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update assistant: {str(e)}")


@app.get("/api/assistant/{assistant_id}")
async def get_assistant_info(
    assistant_id: str,
    vapi_key: Optional[str] = Query(
        None,
        description="VAPI API key. If not provided, uses server's VAPI_API_KEY env variable."
    ),
):
    """Get assistant info and check if Hebrew TTS is enabled."""

    # Use provided API key or fall back to server config
    api_key = vapi_key or vapi_api_key

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="VAPI API key required. Provide 'vapi_key' query parameter or configure VAPI_API_KEY on server."
        )

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{VAPI_BASE_URL}/assistant/{assistant_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Assistant {assistant_id} not found")
            elif response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to get assistant: {response.text}"
                )

            assistant = response.json()

            # Check if using custom voice with deepdub
            voice = assistant.get("voice", {})
            is_hebrew_tts = (
                voice.get("provider") == "custom-voice" and
                "to-speech" in voice.get("server", {}).get("url", "").lower()
            )

            return {
                "id": assistant.get("id"),
                "name": assistant.get("name"),
                "voice": voice,
                "is_hebrew_tts_enabled": is_hebrew_tts,
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get assistant: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)