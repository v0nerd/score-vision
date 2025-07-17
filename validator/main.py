import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List

import aiohttp
import httpx
from dotenv import load_dotenv
from fiber.chain import signatures, fetch_nodes
from fiber.chain.models import Node
from fiber.chain.interface import get_substrate
from fiber.chain.chain_utils import load_hotkey_keypair, load_coldkeypub_keypair
from loguru import logger
from substrateinterface import Keypair
import secrets

# Add project root to Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

from validator.db.operations import DatabaseManager
from validator.config import (
    NETUID, SUBTENSOR_NETWORK, SUBTENSOR_ADDRESS,
    WALLET_NAME, HOTKEY_NAME,
    MIN_MINERS, MAX_MINERS, MIN_STAKE_THRESHOLD,
    CHALLENGE_INTERVAL, CHALLENGE_TIMEOUT, DB_PATH,
    SCORE_THRESHOLD, WEIGHTS_INTERVAL,
    WEIGHT_EVALUATION, WEIGHT_AVAILABILITY, WEIGHT_SPEED,
    OPENAI_API_KEY
)
from validator.challenge.send_challenge import send_challenge
from validator.challenge.challenge_types import (
    ChallengeType, GSRChallenge, GSRResponse, ValidationResult, ChallengeTask
)
from validator.evaluation.evaluation import GSRValidator
from validator.evaluation.set_weights import set_weights
from validator.evaluation.calculate_score import calculate_score
from validator.db.schema import init_db
from validator.evaluation.evaluation_loop import run_evaluation_loop
from validator.utils.api import get_next_challenge

# TODO check why stopped working, only doing availablity check, but not logging anything.
# Load environment variables
validator_dir = Path(__file__).parent
env_path = validator_dir / ".env"
load_dotenv(env_path)

class ChallengeTask:
    def __init__(self, node_id: int, task: asyncio.Task, timestamp: datetime, challenge: GSRChallenge, miner_hotkey: str):
        self.node_id = node_id
        self.task = task
        self.timestamp = timestamp
        self.challenge = challenge
        self.miner_hotkey = miner_hotkey

def get_active_nodes_with_stake() -> list[Node]:
    """Get list of active nodes that meet the stake requirement (less than 100 TAO)."""
    try:
        # Get nodes from chain
        substrate = get_substrate(
            subtensor_network=SUBTENSOR_NETWORK,
            subtensor_address=SUBTENSOR_ADDRESS
        )
        
        nodes = fetch_nodes.get_nodes_for_netuid(substrate, NETUID)
        logger.info(f"Found {len(nodes)} total nodes on chain")
        
        # Filter for active nodes with less than 100 TAO stake
        MAX_STAKE = 100.0  # 100 TAO maximum stake
        active_nodes = [
            node for node in nodes
            if node.stake < MAX_STAKE  # Changed from >= MIN_STAKE_THRESHOLD to < MAX_STAKE
        ]
        
        # Log details about active nodes
        logger.info(f"Found {len(active_nodes)} nodes with stake less than {MAX_STAKE} TAO")
        for node in active_nodes:
            logger.info(f"Active node: {node.hotkey}")
            logger.info(f"  - Node ID: {node.node_id}")
            logger.info(f"  - Stake: {node.stake} TAO")
            logger.info(f"  - IP: {node.ip}")
            logger.info(f"  - Port: {node.port}")
            logger.info(f"  - Last update: {node.last_updated}")
        
        result = active_nodes[:MAX_MINERS] if MAX_MINERS else active_nodes
        logger.info(f"Returning {len(result)} nodes after MAX_MINERS limit of {MAX_MINERS}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to get active nodes: {str(e)}")
        return []

async def process_challenge_results(
    challenge_tasks: List[ChallengeTask],
    db_manager: DatabaseManager,
    validator: GSRValidator,
    keypair: Keypair,
    substrate: Any
) -> None:
    """Process challenge results without blocking."""
    logger.info(f"Processing {len(challenge_tasks)} challenge results")
    
    # Wait for all tasks to complete with timeout
    pending = [task.task for task in challenge_tasks]
    timeout = 3600  # 1 hour timeout
    
    while pending:
        # Wait for the next task to complete, with timeout
        done, pending = await asyncio.wait(
            pending,
            timeout=60,  # Check every minute for completed tasks
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Process completed tasks
        for task in done:
            try:
                response = await task
                # Process the response...
                logger.debug(f"Processed challenge response: {response}")
            except Exception as e:
                logger.error(f"Error processing challenge result: {str(e)}")
        
        # Log status of remaining tasks
        if pending:
            logger.info(f"Still waiting for {len(pending)} challenges to complete")
    
    logger.info("All challenge results processed")

def construct_server_address(node: Node) -> str:
    """Construct server address for a node.
    
    For local development:
    - Nodes register as 0.0.0.1 on the chain (since 127.0.0.1 is not allowed)
    - But we connect to them via 127.0.0.1 locally
    """
    if node.ip == "0.0.0.1":
        # For local development, connect via localhost
        return f"http://127.0.0.1:{node.port}"
    return f"http://{node.ip}:{node.port}"

async def check_miner_availability(
    node: Node,
    client: httpx.AsyncClient,
    db_manager: DatabaseManager,
    hotkey: str
) -> bool:
    """Check if a miner is available and log the result."""
    server_address = construct_server_address(node)
    start_time = time.time()
    
    try:
        headers = {"validator-hotkey": hotkey}
        response = await client.get(f"{server_address}/availability", headers=headers, timeout=5.0)
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        is_available = response.json().get("available", False)
        
        # Log availability check
        db_manager.log_availability_check(
            node_id=node.node_id,
            hotkey=node.hotkey,
            is_available=is_available,
            response_time_ms=response_time
        )
        
        return is_available
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        # Log failed check
        db_manager.log_availability_check(
            node_id=node.node_id,
            hotkey=node.hotkey,
            is_available=False,
            response_time_ms=response_time,
            error=str(e)
        )
        
        logger.warning(f"Failed to check availability for node {node.node_id}: {str(e)}")
        return False

async def get_available_nodes(
    nodes: list[Node],
    client: httpx.AsyncClient,
    db_manager: DatabaseManager,
    hotkey: str
) -> list[Node]:
    """Check availability of all nodes and return available ones."""
    availability_tasks = [
        check_miner_availability(node, client, db_manager, hotkey)
        for node in nodes
    ]
    
    availability_results = await asyncio.gather(*availability_tasks)
    available_nodes = [
        node for node, is_available in zip(nodes, availability_results)
        if is_available
    ]
    
    # If we have more available nodes than MAX_MINERS, randomly select MAX_MINERS
    if len(available_nodes) > MAX_MINERS:
        logger.info(f"Found {len(available_nodes)} available nodes, selecting {MAX_MINERS} randomly")
        available_nodes = secrets.SystemRandom().sample(available_nodes, MAX_MINERS)
    
    return available_nodes

async def weights_update_loop(db_manager: DatabaseManager) -> None:
    """Run the weights update loop on WEIGHTS_INTERVAL."""
    logger.info("Starting weights update loop")
    while True:
        try:
            await set_weights(db_manager)
            logger.info(f"Weights updated successfully, sleeping for {WEIGHTS_INTERVAL}")
            await asyncio.sleep(WEIGHTS_INTERVAL.total_seconds())
        except Exception as e:
            logger.error(f"Error in weights update loop: {str(e)}")
            await asyncio.sleep(WEIGHTS_INTERVAL.total_seconds())

async def periodic_cleanup(db_manager: DatabaseManager, interval_hours: int = 24):
    """
    Periodically clean up old data from the database.
    
    Args:
        db_manager: DatabaseManager instance
        interval_hours: Number of hours between cleanup operations
    """
    while True:
        try:
            logger.info("Starting periodic database cleanup")
            db_manager.cleanup_old_data()
            logger.info("Periodic database cleanup completed")
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {str(e)}")
        
        # Wait for the next cleanup interval
        await asyncio.sleep(interval_hours * 3600)

async def main():
    """Main validator loop."""
    # Load configuration
    load_dotenv()
    
    # Get environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Initialize database manager and validator
    logger.info(f"Initializing database manager with path: {DB_PATH}")
    db_manager = DatabaseManager(DB_PATH)
    validator = GSRValidator(openai_api_key=OPENAI_API_KEY)
    
    # Load validator keys
    try:
        hotkey = load_hotkey_keypair(WALLET_NAME, HOTKEY_NAME)
        coldkey = load_coldkeypub_keypair(WALLET_NAME)
    except Exception as e:
        logger.error(f"Failed to load keys: {str(e)}")
        return

    # Initialize substrate connection
    substrate = get_substrate(
        subtensor_network=SUBTENSOR_NETWORK,
        subtensor_address=SUBTENSOR_ADDRESS
    )

    # Initialize HTTP client with long timeout
    async with httpx.AsyncClient(timeout=CHALLENGE_TIMEOUT.total_seconds()) as client:
        active_challenge_tasks = []  # Track active challenges
        
        # Start evaluation loop as a separate task
        evaluation_task = asyncio.create_task(
            run_evaluation_loop(
                db_path=DB_PATH,
                openai_api_key=OPENAI_API_KEY,
                validator_hotkey=hotkey.ss58_address,
                batch_size=10,
                sleep_interval=60
            )
        )
        
        # Start weights update loop as a separate task
        weights_task = asyncio.create_task(
            weights_update_loop(db_manager)
        )
        
        # Start the periodic cleanup task
        cleanup_task = asyncio.create_task(periodic_cleanup(db_manager))
        
        try:
            # Main challenge loop
            while True:
                try:
                    # Clean up completed challenge tasks
                    active_challenge_tasks = [
                        task for task in active_challenge_tasks 
                        if not task.task.done()
                    ]
                    
                    # Get active nodes with sufficient stake
                    active_nodes = get_active_nodes_with_stake()
                    num_active = len(active_nodes)
                    
                    if num_active < MIN_MINERS:
                        logger.warning(f"Only {num_active} active nodes with sufficient stake (minimum {MIN_MINERS} required)")
                        await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())
                        continue

                    logger.info(f"Found {num_active} active nodes with sufficient stake")
                    
                    # Check availability of nodes
                    available_nodes = await get_available_nodes(active_nodes, client, db_manager, hotkey.ss58_address)
                    num_available = len(available_nodes)
                    
                    if num_available < MIN_MINERS:
                        logger.warning(f"Only {num_available} nodes are available (minimum {MIN_MINERS} required)")
                        await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())
                        continue

                    logger.info(f"Processing {num_available} available nodes")

                    # Generate and send challenges
                    challenge_time = time.time()
                    new_challenge_tasks = []
                    
                    # Fetch next challenge from API
                    challenge_data = await get_next_challenge(hotkey.ss58_address)
                    if not challenge_data:
                        logger.warning("No challenge available from API, sleeping...")
                        await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())
                        continue

                    logger.info(f"Got challenge from API: task_id={challenge_data['task_id']}")
                    
                    for node in available_nodes:
                        # Create challenge
                        challenge = GSRChallenge(
                            challenge_id=challenge_data['task_id'],
                            type=ChallengeType.GSR,
                            created_at=datetime.now(timezone.utc),
                            video_url=challenge_data['video_url']
                        )
                        
                        task = asyncio.create_task(
                            send_challenge(
                                challenge=challenge,
                                server_address=construct_server_address(node),
                                hotkey=node.hotkey,
                                keypair=hotkey,
                                node_id=node.node_id,
                                db_manager=db_manager,
                                client=client
                            )
                        )
                        
                        challenge_task = ChallengeTask(
                            node_id=node.node_id,
                            task=task,
                            timestamp=datetime.now(timezone.utc),
                            challenge=challenge,
                            miner_hotkey=node.hotkey
                        )
                        new_challenge_tasks.append(challenge_task)
                    
                    # Add new challenges to active tasks
                    active_challenge_tasks.extend(new_challenge_tasks)
                    
                    # Process any completed challenges
                    await process_challenge_results(
                        new_challenge_tasks,
                        db_manager,
                        validator,
                        hotkey,
                        substrate
                    )

                    # Log status
                    num_active_challenges = len(active_challenge_tasks)
                    if num_active_challenges > 0:
                        logger.info(f"Currently tracking {num_active_challenges} active challenges")

                    # Sleep until next challenge interval
                    await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())
        finally:
            # Cancel evaluation and weights loops
            evaluation_task.cancel()
            weights_task.cancel()
            cleanup_task.cancel()
            try:
                await asyncio.gather(evaluation_task, weights_task, cleanup_task, return_exceptions=True)
            except asyncio.CancelledError:
                pass

    # Cleanup
    if db_manager:
        db_manager.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
