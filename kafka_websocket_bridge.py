# ============================================================================
# 5. KAFKA CONSUMER FOR WEBSOCKET - kafka_websocket_bridge.py
# ============================================================================

import asyncio
import websockets
import json
from kafka import KafkaConsumer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaWebSocketBridge:
    def __init__(self):
        self.consumers = {}
        self.connected_clients = set()
        
    async def kafka_to_websocket(self, websocket, path):
        """
        Bridge Kafka messages to WebSocket clients
        """
        self.connected_clients.add(websocket)
        logger.info(f"New client connected. Total clients: {len(self.connected_clients)}")
        
        # Create Kafka consumer for this client
        consumer = KafkaConsumer(
            'processed-transactions',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='websocket-consumer-group'
        )
        
        try:
            for message in consumer:
                if websocket not in self.connected_clients:
                    break
                    
                transaction = message.value
                await websocket.send(json.dumps(transaction))
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        finally:
            self.connected_clients.remove(websocket)
            consumer.close()
            logger.info(f"Client removed. Total clients: {len(self.connected_clients)}")
    
    async def start_server(self, host='localhost', port=8765):
        """
        Start WebSocket server
        """
        async with websockets.serve(self.kafka_to_websocket, host, port):
            logger.info(f"WebSocket server started on ws://{host}:{port}")
            await asyncio.Future()  # Run forever

if __name__ == "__main__":
    bridge = KafkaWebSocketBridge()
    asyncio.run(bridge.start_server())