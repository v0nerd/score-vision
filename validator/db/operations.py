import json
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from ..challenge.challenge_types import (
    ChallengeType,
    GSRChallenge,
    GSRResponse,
    ValidationResult
)
from .schema import check_db_initialized, init_db

from fiber.logging_utils import get_logger

logger = get_logger(__name__)

class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        
        # Initialize database if needed
        if not check_db_initialized(str(db_path)):
            logger.info(f"Initializing new database at {db_path}")
            init_db(str(db_path))
            
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to database at {db_path}")
        
    def close(self):
        if self.conn:
            self.conn.close()
        
    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
        
    def store_challenge(self, challenge_id: int, challenge_type: str, video_url: str, task_name: str = None) -> None:
        """Store a new challenge in the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO challenges (
                    challenge_id, type, video_url, created_at, task_name
                )
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
            """, (
                challenge_id,
                challenge_type,
                video_url,
                task_name
            ))
            
            if cursor.rowcount == 0:
                logger.debug(f"Challenge {challenge_id} already exists in database")
            else:
                logger.info(f"Stored new challenge {challenge_id} in database")
            
            conn.commit()
            
        finally:
            conn.close()
            
    def assign_challenge(self, challenge_id: str, miner_hotkey: str, node_id: int) -> None:
        """Assign a challenge to a miner"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO challenge_assignments (
                    challenge_id,
                    miner_hotkey,
                    node_id,
                    status
                )
                VALUES (?, ?, ?, 'assigned')
            """, (
                challenge_id,
                miner_hotkey,
                node_id
            ))
            
            conn.commit()
            
        finally:
            conn.close()

    def mark_challenge_sent(self, challenge_id: str, miner_hotkey: str) -> None:
        """Mark a challenge as sent to a miner"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE challenge_assignments
                SET status = 'sent', sent_at = CURRENT_TIMESTAMP
                WHERE challenge_id = ? AND miner_hotkey = ?
            """, (challenge_id, miner_hotkey))
            
            conn.commit()
            
        finally:
            conn.close()

    def mark_challenge_failed(self, challenge_id: str, miner_hotkey: str) -> None:
        """Mark a challenge as failed for a miner"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE challenge_assignments
                SET status = 'failed'
                WHERE challenge_id = ? AND miner_hotkey = ?
            """, (challenge_id, miner_hotkey))
            
            conn.commit()
            
        finally:
            conn.close()

    def store_response(
        self, 
        challenge_id: str, 
        miner_hotkey: str, 
        response: GSRResponse,
        node_id: int,
        processing_time: float = 0.0
    ) -> int:
        """Store a miner's response to a challenge"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            now = datetime.utcnow()
            
            # Convert response to dict and handle frames data
            response_dict = response.to_dict()
            
            # Store response
            cursor.execute("""
                INSERT INTO responses (
                    challenge_id,
                    miner_hotkey,
                    node_id,
                    processing_time,
                    response_data,
                    received_at,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                challenge_id,
                miner_hotkey,
                node_id,
                processing_time,
                json.dumps(response_dict),
                now,
                now
            ))
            
            response_id = cursor.lastrowid
            
            # Mark challenge as completed in challenge_assignments
            cursor.execute("""
                UPDATE challenge_assignments
                SET status = 'completed',
                    completed_at = ?
                WHERE challenge_id = ? AND miner_hotkey = ?
            """, (now, challenge_id, miner_hotkey))
            
            conn.commit()
            return response_id
            
        finally:
            conn.close()
            
    def store_response_score(
        self, 
        response_id: int, 
        validation_result: ValidationResult, 
        validator_hotkey: str = None,
        availability_score: float = 1.0,
        speed_score: float = 1.0,
        total_score: float = None
    ) -> None:
        """
        Store the score for a response
        
        Args:
            response_id: ID of the response
            validation_result: ValidationResult object with evaluation score
            validator_hotkey: Validator's public key
            availability_score: Score based on miner availability (0-1)
            speed_score: Score based on processing speed (0-1)
            total_score: Optional total weighted score (if None, will use validation_result.score)
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            now = datetime.utcnow()
            
            # Get response details
            cursor.execute("""
                SELECT challenge_id, miner_hotkey, processing_time
                FROM responses
                WHERE response_id = ?
            """, (response_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Response {response_id} not found")
                
            challenge_id, miner_hotkey, processing_time = row
            
            # Calculate total score if not provided
            if total_score is None:
                total_score = (
                    validation_result.score * 0.6 +  # 60% evaluation
                    availability_score * 0.3 +       # 30% availability
                    speed_score * 0.1                # 10% speed
                )
            
            # Update response with score and mark as evaluated
            cursor.execute("""
                UPDATE responses
                SET score = ?,
                    evaluated = TRUE,
                    evaluated_at = ?
                WHERE response_id = ?
            """, (total_score, now, response_id))
            
            # Store detailed score breakdown
            cursor.execute("""
                INSERT INTO response_scores (
                    response_id,
                    challenge_id,
                    miner_hotkey,
                    validator_hotkey,
                    evaluation_score,
                    availability_score,
                    speed_score,
                    total_score,
                    response_time,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                response_id,
                challenge_id,
                miner_hotkey,
                validator_hotkey,
                validation_result.score,
                availability_score,
                speed_score,
                total_score,
                processing_time,
                now
            ))
            
            conn.commit()
            
        finally:
            conn.close()

    def store_frame_evaluation(
        self,
        response_id: int,
        challenge_id: str,
        miner_hotkey: str,
        node_id: int,
        frame_id: int,
        frame_timestamp: float,
        frame_score: float,
        raw_frame_path: str,
        annotated_frame_path: str,
        vlm_response: dict,
        feedback: str
    ) -> None:
        """Store a frame evaluation result"""
        if response_id is None:
            raise ValueError("response_id is required for frame evaluation")
            
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO frame_evaluations (
                    response_id,
                    challenge_id,
                    miner_hotkey,
                    node_id,
                    frame_id,
                    frame_timestamp,
                    frame_score,
                    raw_frame_path,
                    annotated_frame_path,
                    vlm_response,
                    feedback,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                response_id,
                challenge_id,
                miner_hotkey,
                node_id,
                frame_id,
                frame_timestamp,
                frame_score,
                raw_frame_path,
                annotated_frame_path,
                json.dumps(vlm_response),
                feedback
            ))
            
            conn.commit()
            logger.info(f"Stored frame evaluation for response {response_id}, frame {frame_id}")
            
        except Exception as e:
            logger.error(f"Error storing frame evaluation: {str(e)}")
            raise
        finally:
            conn.close()

    def get_miner_scores(self, hours: int = 24) -> Dict[str, float]:
        """Get average miner scores from the last N hours"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Calculate cutoff time
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Get average scores for each miner
            cursor.execute("""
                SELECT 
                    miner_hotkey,
                    AVG(total_score) as avg_score
                FROM response_scores
                WHERE created_at >= ?
                GROUP BY miner_hotkey
            """, (cutoff_time,))
            
            # Convert to dictionary
            scores = {}
            for row in cursor.fetchall():
                miner_hotkey, avg_score = row
                scores[miner_hotkey] = float(avg_score) if avg_score is not None else 0.0
            
            logger.info(f"Retrieved scores for {len(scores)} miners from the last {hours} hours")
            return scores
            
        finally:
            conn.close()

    def get_challenge(self, challenge_id: str) -> Optional[Dict]:
        """Get a challenge from the database by ID"""
        conn = self.get_connection()
        with conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.*, ca.sent_at 
                FROM challenges c
                LEFT JOIN challenge_assignments ca ON c.challenge_id = ca.challenge_id
                WHERE c.challenge_id = ?
            """, (challenge_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None

    def update_miner_availability(self, miner_hotkey: str, node_id: int, is_available: bool = True) -> None:
        """Record a new availability check for a miner"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO miner_availability 
                (miner_hotkey, node_id, is_available)
                VALUES (?, ?, ?)
            """, (miner_hotkey, node_id, is_available))
            conn.commit()

    def get_frame_evaluations(
        self,
        challenge_id: str = None,
        miner_hotkey: str = None,
        node_id: int = None,
        response_id: int = None
    ) -> List[Dict]:
        """Get frame evaluations with optional filters"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            sql = "SELECT * FROM frame_evaluations WHERE 1=1"
            params = []
            
            if challenge_id:
                sql += " AND challenge_id = ?"
                params.append(challenge_id)
            if miner_hotkey:
                sql += " AND miner_hotkey = ?"
                params.append(miner_hotkey)
            if node_id:
                sql += " AND node_id = ?"
                params.append(node_id)
            if response_id:
                sql += " AND response_id = ?"
                params.append(response_id)
                
            sql += " ORDER BY frame_timestamp ASC"
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(row)
                if result['vlm_response']:
                    result['vlm_response'] = json.loads(result['vlm_response'])
                results.append(result)
                
            return results
            
        finally:
            conn.close()

    def get_processing_time_stats(self, challenge_id: str) -> Dict[str, float]:
        """
        Get processing time statistics for all responses to the same challenge.
        Processing times are in seconds.
        
        Args:
            challenge_id: The challenge ID to get stats for
            
        Returns:
            Dict with avg_time, min_time, max_time in seconds
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    AVG(processing_time) as avg_time,
                    MIN(processing_time) as min_time,
                    MAX(processing_time) as max_time
                FROM responses
                WHERE challenge_id = ?
                AND processing_time > 0
            """, (str(challenge_id),))
            
            row = cursor.fetchone()
            if row:
                return {
                    'avg_time': row[0] or 100.0,  # Default to 5 seconds if no data
                    'min_time': row[1] or 5.0,  # Minimum 1 second
                    'max_time': row[2] or 200.0  # Maximum 10 seconds
                }
            return {
                'avg_time': 100.0,  # Default values in seconds
                'min_time': 5.0,
                'max_time': 200.0
            }
            
        finally:
            conn.close()

    def get_completed_tasks(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get completed tasks from the last N hours"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    ca.challenge_id,
                    ca.node_id,
                    ca.miner_hotkey,
                    ca.sent_at,
                    ca.received_at,
                    ca.task_returned_data,
                    c.type,
                    c.task_name
                FROM challenge_assignments ca
                JOIN challenges c ON ca.challenge_id = c.challenge_id
                WHERE ca.status = 'completed'
                AND ca.received_at >= datetime('now', ? || ' hours')
                ORDER BY ca.received_at DESC
            """, (-hours,))
            
            rows = cursor.fetchall()
            tasks = []
            for row in rows:
                task = {
                    'task_id': row[0],
                    'node_id': row[1],
                    'miner_hotkey': row[2],
                    'sent_at': row[3],
                    'received_at': row[4],
                    'task_returned_data': row[5],
                    'type': row[6],
                    'task_name': row[7]
                }
                tasks.append(task)
                
            return tasks
            
        finally:
            conn.close()

    def get_challenges_with_unevaluated_responses(self) -> List[Dict]:
        """Get challenges that have responses without evaluations"""
        query = """
        SELECT DISTINCT c.*
        FROM challenges c
        JOIN responses r ON c.challenge_id = r.challenge_id
        LEFT JOIN response_scores rs ON r.response_id = rs.response_id
        WHERE rs.response_id IS NULL
        """
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_unevaluated_responses(self, challenge_id: str) -> List[Dict]:
        """Get responses for a challenge that haven't been evaluated yet"""
        query = """
        SELECT 
            r.response_id,
            r.challenge_id,
            r.node_id,
            r.miner_hotkey,
            r.processing_time,
            r.response_data
        FROM responses r
        LEFT JOIN response_scores rs ON r.response_id = rs.response_id
        WHERE r.challenge_id = ? AND rs.response_id IS NULL
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (challenge_id,))
            rows = cursor.fetchall()
            
            responses = []
            for row in rows:
                # Create GSRResponse compatible dict
                gsr_response = {
                    'challenge_id': row[1],
                    'node_id': row[2],
                    'miner_hotkey': row[3],
                    'processing_time': row[4],
                    'frames': {}
                }
                
                # Parse response_data JSON and extract frames
                if row[5]:  # response_data
                    try:
                        response_data = json.loads(row[5])
                        # The frames data is nested inside response_data['frames']
                        frames_data = response_data.get('frames', {})
                        if isinstance(frames_data, dict):
                            gsr_response['frames'] = frames_data
                        else:
                            logger.warning(f"Frames data in response {row[0]} is not a dictionary: {type(frames_data)}")
                            gsr_response['frames'] = {}
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse response data for response {row[0]}: {e}")
                        gsr_response['frames'] = {}
                
                # Add response_id as a separate field
                gsr_response['response_id'] = row[0]
                responses.append(gsr_response)
            
            return responses

    def get_challenge(self, challenge_id: str) -> Optional[Dict]:
        """Get challenge details"""
        conn = self.get_connection()
        with conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.*, ca.sent_at 
                FROM challenges c
                LEFT JOIN challenge_assignments ca ON c.challenge_id = ca.challenge_id
                WHERE c.challenge_id = ?
            """, (challenge_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
            
    def get_challenge_frames(self, challenge_id: str) -> List[int]:
        """Get frame numbers selected for a challenge"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT frame_number
                FROM challenge_frames
                WHERE challenge_id = ?
                ORDER BY frame_number
            """, (challenge_id,))
            
            return [row[0] for row in cursor.fetchall()]
            
        finally:
            conn.close()

    def store_challenge_frames(self, challenge_id: str, frame_numbers: List[int]) -> None:
        """Store selected frame numbers for a challenge"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.executemany("""
                INSERT INTO challenge_frames (challenge_id, frame_number)
                VALUES (?, ?)
            """, [(challenge_id, frame_num) for frame_num in frame_numbers])
            
            conn.commit()
            
        finally:
            conn.close()

    def get_frame_scores(self, challenge_id: str, response_id: int) -> List[float]:
        """Get all frame scores for a response"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT frame_score
                FROM frame_evaluations
                WHERE challenge_id = ? AND response_id = ?
                ORDER BY frame_id
            """, (challenge_id, response_id))
            
            return [row[0] for row in cursor.fetchall()]
            
        finally:
            conn.close()

    def update_response_score(self, response_id: int, score: float) -> None:
        """Update the overall score for a response and mark it as evaluated"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE responses
                SET score = ?, evaluated = TRUE, evaluated_at = ?
                WHERE id = ?
            """, (score, datetime.utcnow(), response_id))
            
            conn.commit()
            
        finally:
            conn.close()

    def get_miner_scores(self) -> List[Dict]:
        """Get overall scores for all miners from evaluated responses"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    r.miner_hotkey,
                    AVG(r.score) as overall_score,
                    COUNT(*) as total_responses,
                    AVG(r.processing_time) as avg_processing_time
                FROM responses r
                WHERE r.evaluated = TRUE
                GROUP BY r.miner_hotkey
            """)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        finally:
            conn.close()

    def get_node_scores(self) -> Dict[str, Dict[str, float]]:
        """Get average scores for each node over the last 24 hours with component breakdowns"""
        query = """
        WITH performance_scores AS (
            SELECT 
                r.node_id,
                r.miner_hotkey,
                AVG(rs.evaluation_score) as performance_score,
                AVG(CASE 
                    WHEN r.processing_time <= 30 THEN 1.0
                    WHEN r.processing_time <= 60 THEN 0.8
                    WHEN r.processing_time <= 120 THEN 0.5
                    ELSE 0.2
                END) as speed_score,
                COUNT(*) as response_count
            FROM responses r
            JOIN response_scores rs ON r.response_id = rs.response_id
            WHERE r.received_at >= datetime('now', '-24 hours')
            GROUP BY r.node_id, r.miner_hotkey
            HAVING response_count > 0
        )
        SELECT 
            p.node_id,
            p.miner_hotkey,
            p.performance_score,
            p.speed_score,
            p.response_count
        FROM performance_scores p
        WHERE p.performance_score IS NOT NULL
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            
            scores = {}
            for row in rows:
                node_id = str(row[0])
                hotkey = row[1]
                perf_score = row[2]
                speed_score = row[3]
                response_count = row[4]
                
                # Get availability score
                avail_score = self.get_availability_score(int(node_id))
                
                # Calculate final score with weights
                final_score = (
                    perf_score * 0.6 +  # Performance weight
                    speed_score * 0.2 +  # Speed weight
                    avail_score * 0.2    # Availability weight
                )
                
                scores[node_id] = {
                    'miner_hotkey': hotkey,
                    'performance_score': perf_score,
                    'speed_score': speed_score,
                    'availability_score': avail_score,
                    'final_score': final_score,
                    'response_count': response_count
                }
            
            logger.info(f"Calculated scores for {len(scores)} nodes")
            for node_id, score_data in scores.items():
                logger.debug(f"Node {node_id} scores:")
                logger.debug(f"  Performance: {score_data['performance_score']:.3f}")
                logger.debug(f"  Speed: {score_data['speed_score']:.3f}")
                logger.debug(f"  Availability: {score_data['availability_score']:.3f}")
                logger.debug(f"  Final Score: {score_data['final_score']:.3f}")
                logger.debug(f"  Responses: {score_data['response_count']}")
            
            return scores

    def get_availability_score(self, node_id: int) -> float:
        """Get availability score for a node"""
        query = """
        SELECT COUNT(CASE WHEN is_available THEN 1 END) * 1.0 / COUNT(*) as availability_score
        FROM availability_checks
        WHERE node_id = ? AND checked_at >= datetime('now', '-24 hours')
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (node_id,))
            row = cursor.fetchone()
            return row[0] if row and row[0] is not None else 0.0

    async def create_challenge(self, video_url: str, external_task_id: int) -> Optional[int]:
        """
        Create a new challenge in the database.
        
        Args:
            video_url: URL of the video for the challenge
            external_task_id: Task ID from the external API
            
        Returns:
            challenge_id if successful, None otherwise
        """
        try:
            query = """
            INSERT INTO challenges (video_url, external_task_id, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            RETURNING challenge_id
            """
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (video_url, external_task_id))
                row = cursor.fetchone()
                conn.commit()
                
                if row:
                    challenge_id = row[0]
                    logger.info(f"Created challenge {challenge_id} for external task {external_task_id}")
                    return challenge_id
                    
                return None
                
        except Exception as e:
            logger.error(f"Error creating challenge: {str(e)}")
            return None

    def has_challenge_assignment(self, challenge_id: str, miner_hotkey: str) -> bool:
        """Check if a miner has already been assigned a challenge"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 1
                FROM challenge_assignments
                WHERE challenge_id = ? AND miner_hotkey = ?
            """, (challenge_id, miner_hotkey))
            
            return cursor.fetchone() is not None
            
        finally:
            conn.close()

    def log_availability_check(
        self,
        node_id: int,
        hotkey: str,
        is_available: bool,
        response_time_ms: float,
        error: Optional[str] = None
    ) -> None:
        """Log an availability check for a miner."""
        query = """
        INSERT INTO availability_checks 
        (node_id, hotkey, checked_at, is_available, response_time_ms, error)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        with self.conn:
            self.conn.execute(
                query,
                (node_id, hotkey, datetime.utcnow(), is_available, response_time_ms, error)
            )

    def get_recent_availability(
        self,
        node_id: int,
        minutes: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent availability checks for a node."""
        query = """
        SELECT * FROM availability_checks
        WHERE node_id = ?
        AND checked_at >= datetime('now', ?)
        ORDER BY checked_at DESC
        """
        with self.conn:
            cursor = self.conn.execute(query, (node_id, f'-{minutes} minutes'))
            return [dict(row) for row in cursor.fetchall()]

    def get_availability_stats(
        self,
        node_id: int,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get availability statistics for a node."""
        query = """
        SELECT 
            COUNT(*) as total_checks,
            SUM(CASE WHEN is_available = 1 THEN 1 ELSE 0 END) as available_count,
            AVG(CASE WHEN is_available = 1 THEN response_time_ms ELSE NULL END) as avg_response_time
        FROM availability_checks
        WHERE node_id = ?
        AND checked_at >= datetime('now', ?)
        """
        with self.conn:
            cursor = self.conn.execute(query, (node_id, f'-{hours} hours'))
            stats = dict(cursor.fetchone())
            stats['availability_rate'] = (
                stats['available_count'] / stats['total_checks']
                if stats['total_checks'] > 0 else 0
            )
            return stats

    def cleanup_old_data(self, days: int = 7) -> None:
        """
        Remove data older than the specified number of days from various tables.
        
        Args:
            days: Number of days to keep data for. Default is 7.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Define tables and their timestamp columns
            tables_to_clean = [
                ("responses", "received_at"),
                ("challenge_assignments", "completed_at"),
                ("frame_evaluations", "created_at"),
                ("response_scores", "created_at"),
                ("availability_checks", "checked_at")
            ]
            
            for table, timestamp_column in tables_to_clean:
                query = f"""
                DELETE FROM {table}
                WHERE {timestamp_column} < datetime('now', ?)
                """
                cursor.execute(query, ('-{0} days'.format(days), ))
                deleted_rows = cursor.rowcount
                logger.info(f"Deleted {deleted_rows} rows from {table} older than {days} days")
            
            # Clean up challenges that are no longer referenced
            cursor.execute("""
                DELETE FROM challenges
                WHERE challenge_id NOT IN (
                    SELECT DISTINCT challenge_id FROM responses
                    UNION
                    SELECT DISTINCT challenge_id FROM challenge_assignments
                )
                AND created_at < datetime('now', '-{days} days')
            """)
            deleted_challenges = cursor.rowcount
            logger.info(f"Deleted {deleted_challenges} orphaned challenges older than {days} days")
            
            conn.commit()
            logger.info(f"Database cleanup completed for data older than {days} days")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error during database cleanup: {str(e)}")
        finally:
            conn.close()
