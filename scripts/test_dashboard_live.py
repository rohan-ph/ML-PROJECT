import asyncio
import websockets
import json
import requests
import time

async def test_ws():
    uri = 'ws://127.0.0.1:8000/ws/stream'
    async with websockets.connect(uri) as ws:
        # Start stream with uploaded video
        res = requests.post('http://127.0.0.1:8000/api/start_stream', json={'source': 'data/sample_videos/V_10.mp4'})
        print('Start Stream response:', res.json())
        
        start_t = time.time()
        packets = 0
        evidence_received = False
        while time.time() - start_t < 15:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                data = json.loads(msg)
                msg_type = data.get('type')
                if msg_type == 'telemetry':
                    packets += 1
                    t = data['payload']
                    if packets % 15 == 1 or t['violence_score'] > 0.5:
                        print(f"Frame {t['frame_idx']}: Score={t['violence_score']:.3f}, State={t['incident_state']}, Spikes={t['total_spikes']}, HasFrame={'frame_jpeg' in data}")
                elif msg_type == 'evidence_ready':
                    print('>>> EVIDENCE_READY received! Clip ID:', data['payload'].get('clip_id'), 'Hash:', data['payload'].get('sha256_hash'))
                    evidence_received = True
                elif msg_type == 'stream_ended':
                    print('>>> STREAM_ENDED event received!')
                    break
            except asyncio.TimeoutError:
                print('Timeout waiting for WS packet')
                break
        print(f'Done test: received {packets} telemetry packets, evidence_received={evidence_received}')

if __name__ == '__main__':
    asyncio.run(test_ws())
