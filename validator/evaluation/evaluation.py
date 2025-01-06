import os
import json
import base64
import httpx
import tempfile
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import asyncio
from openai import OpenAI
from fiber.logging_utils import get_logger
from validator.challenge.challenge_types import (
    GSRResponse,
    GSRChallenge,
    ValidationResult
)
from validator.config import FRAMES_TO_VALIDATE
from validator.evaluation.prompts import COUNT_PROMPT, VALIDATION_PROMPT
import secrets

logger = get_logger(__name__)

# Define class mappings
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Define colors for annotations
COLORS = {
    "player": (0, 255, 0),      # Green boxes
    "goalkeeper": (0, 0, 255),  # Red boxes
    "referee": (255, 0, 0),     # Blue boxes
    "ball": (0, 255, 255),      # Yellow boxes
    "keypoint": (255, 0, 255)   # Bright pink
}

def optimize_coordinates(coords: List[float]) -> List[float]:
    """Round coordinates to 2 decimal places to reduce data size."""
    return [round(float(x), 2) for x in coords]

def filter_keypoints(keypoints: List[List[float]]) -> List[List[float]]:
    """Filter out keypoints with zero coordinates and round remaining to 2 decimal places."""
    return [optimize_coordinates(kp) for kp in keypoints if not (kp[0] == 0 and kp[1] == 0)]

class GSRValidator:
    def __init__(self, openai_api_key: str, validator_hotkey: str = None):
        """Initialize validator with OpenAI API key for frame validation"""
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for frame validation")
        self.client = OpenAI(api_key=openai_api_key)
        self.validator_hotkey = validator_hotkey
        self.labels = ["player", "goalkeeper", "referee", "ball"]
        self._video_cache = {}  # Cache of downloaded videos
        logger.info("GSRValidator initialized - will perform video frame validation")

    def select_random_frames(self, video_path: Path, num_frames: int = None) -> List[int]:
        """Select random frames from video"""
        if num_frames is None:
            num_frames = FRAMES_TO_VALIDATE
            
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Select random frames, ensuring they're not too close to start/end
        buffer = min(30, total_frames // 10)  # 10% or 30 frames buffer
        frame_range = range(buffer, total_frames - buffer)
        frames = secrets.SystemRandom().sample(frame_range, min(num_frames, len(frame_range)))
        frames.sort()  # Keep frames in order
        return frames

    def draw_annotations(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """Draw annotations on frame with correct colors."""
        annotated_frame = frame.copy()
        
        # Draw objects
        for obj in detections.get("objects", []):
            bbox = obj["bbox"]
            class_id = obj["class_id"]
            
            if class_id == BALL_CLASS_ID:
                color = COLORS["ball"]
            elif class_id == GOALKEEPER_CLASS_ID:
                color = COLORS["goalkeeper"]
            elif class_id == REFEREE_CLASS_ID:
                color = COLORS["referee"]
            else:  # PLAYER_CLASS_ID
                color = COLORS["player"]
                
            cv2.rectangle(
                annotated_frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2
            )
        
        # Draw keypoints
        for point in detections.get("keypoints", []):
            if point[0] != 0 and point[1] != 0:  # Only draw non-zero keypoints
                cv2.circle(
                    annotated_frame,
                    (int(point[0]), int(point[1])),
                    5,
                    COLORS["keypoint"],
                    -1
                )
        
        return annotated_frame

    def get_reference_counts(self, frame: np.ndarray) -> Dict:
        """Get reference counts of objects in frame using VLM"""
        encoded = self.encode_image(frame)
        
        messages = [
            {"role": "system", "content": "You are an expert at counting objects in soccer match frames."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": COUNT_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}}
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=500,
                temperature=0.2
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            try:
                result = json.loads(content)
                logger.info(f"Reference counts: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {content}")
                logger.error(f"JSON parse error: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error getting reference counts: {str(e)}")
            return None

    def validate_bbox_coordinates(self, bbox: List[float], frame_shape: Tuple[int, int], class_id: int) -> Optional[List[int]]:
        """
        Validate and correct bounding box coordinates.
        Returns None if bbox is invalid or out of frame.
        """
        try:
            height, width = frame_shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Basic sanity checks
            if x2 <= x1 or y2 <= y1:
                return None
                
            # Clamp coordinates to frame dimensions
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # Check if bbox is too small after clamping
            if class_id == BALL_CLASS_ID:
                # For ball, allow very small bounding boxes (minimum 1x1)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    return None
            else:
                # For other objects, keep a slightly larger minimum size
                if x2 - x1 < 5 or y2 - y1 < 5:
                    return None
                
            return [x1, y1, x2, y2]
        except Exception as e:
            logger.error(f"Error validating bbox coordinates: {str(e)}")
            return None

    def filter_detections(self, detections: Dict, frame_shape: Tuple[int, int]) -> Dict:
        """
        Filter and validate detections early in the pipeline.
        """
        filtered = {
            "objects": [],
            "keypoints": detections.get("keypoints", [])
        }
        
        # Filter objects
        for obj in detections.get("objects", []):
            bbox = self.validate_bbox_coordinates(obj["bbox"], frame_shape, obj["class_id"])
            if bbox:
                filtered["objects"].append({
                    **obj,
                    "bbox": bbox
                })
            else:
                logger.debug(f"Filtered out object: {obj}")
        
        logger.debug(f"Filtered {len(detections.get('objects', []))} objects to {len(filtered['objects'])}")
        return filtered

    def validate_bbox_content(self, image: np.ndarray, expected_class: str) -> float:
        """Validate bbox content with GPT-4V."""
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            return 0.0
        
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"This image is supposed to contain a {expected_class} from a soccer game. "
                               f"On a scale of 0.0 to 1.0, what is the probability that this image contains "
                               f"a {expected_class}? Respond with ONLY the numerical probability."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=10
            )
            
            probability_text = response.choices[0].message.content.strip()
            try:
                probability = float(probability_text)
                return max(0.0, min(1.0, probability))
            except ValueError:
                logger.error(f"Failed to parse probability: {probability_text}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in bbox validation: {str(e)}")
            return 0.0

    async def validate_keypoints(self, frame: np.ndarray, keypoints: list, frame_idx: int) -> float:
        """Validate keypoints with GPT-4V."""
        # Create visualization
        keypoint_frame = frame.copy()
        for point in keypoints:
            if point[0] != 0 and point[1] != 0:
                cv2.circle(keypoint_frame, (int(point[0]), int(point[1])), 5, COLORS["keypoint"], -1)
        
        # Prepare images
        reference_path = Path(__file__).parent / "pitch-keypoints.jpg"
        reference_image = cv2.imread(str(reference_path))
        
        success, ref_buffer = cv2.imencode('.jpg', reference_image)
        success, kp_buffer = cv2.imencode('.jpg', keypoint_frame)
        
        if not success:
            return 0.0
        
        ref_base64 = base64.b64encode(ref_buffer).decode('utf-8')
        kp_base64 = base64.b64encode(kp_buffer).decode('utf-8')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "I will show you two images:\n"
                            "1. A reference image showing the correct keypoint placements on a soccer field.\n"
                            "2. A prediction image showing keypoints placed on a similar field.\n\n"
                            "The keypoints (pink dots) must mark:\n"
                            "- Corners of the pitch\n"
                            "- Penalty spots\n"
                            "- Center spot, center circle intersections\n"
                            "- Intersections for 18-yard boxes\n\n"
                            "Rate how accurately the predicted keypoints match the reference image keypoints on a scale from 0.0 to 1.0, "
                            "with 1.0 meaning perfectly placed.\n\n"
                            "Criteria:\n"
                            "- If every keypoint is within 5% positional error (based on field dimensions), the score should be around 0.95.\n"
                            "- If keypoints are very far off (more than 10% error), the score should be around 0.3 or lower.\n"
                            "- For minor but noticeable errors (around 5-10% off), scale the score proportionally (0.5 to 0.8 range).\n\n"
                            "The response should be ONLY the numeric score, no additional text.\n"
                        )
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{kp_base64}"}}
                ]
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                logger.error(f"Failed to parse keypoint score: {score_text}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in keypoint validation: {str(e)}")
            return 0.0

    def compare_with_reference_counts(self, reference_counts: Dict[str, int], validation_results: Dict) -> Dict:
        """Compare high-confidence detections with reference counts."""
        high_confidence_counts = {
            "player": 0,
            "goalkeeper": 0,
            "referee": 0,
            "ball": 0
        }
        
        # Count high-confidence objects by type
        for obj in validation_results.get("objects", []):
            if obj.get("probability", 0) >= 0.9:
                if obj.get("class_id") == BALL_CLASS_ID:
                    high_confidence_counts["ball"] += 1
                elif obj.get("class_id") == GOALKEEPER_CLASS_ID:
                    high_confidence_counts["goalkeeper"] += 1
                elif obj.get("class_id") == REFEREE_CLASS_ID:
                    high_confidence_counts["referee"] += 1
                else:  # regular player
                    high_confidence_counts["player"] += 1
        
        logger.info(f"Reference counts: {reference_counts}")
        logger.info(f"High confidence counts: {high_confidence_counts}")
        
        # Calculate match percentages
        count_matches = {
            "player": min(high_confidence_counts["player"], reference_counts["player"]) / max(reference_counts["player"], 1),
            "goalkeeper": min(high_confidence_counts["goalkeeper"], reference_counts["goalkeeper"]) / max(reference_counts["goalkeeper"], 1),
            "referee": min(high_confidence_counts["referee"], reference_counts["referee"]) / max(reference_counts["referee"], 1),
            "ball": min(high_confidence_counts["ball"], reference_counts.get("soccer ball", 0)) / max(reference_counts.get("soccer ball", 0), 1)
        }
        
        # Calculate overall match score
        total_weight = sum([
            reference_counts["player"],
            reference_counts["goalkeeper"],
            reference_counts["referee"],
            reference_counts.get("soccer ball", 0)
        ])
        
        if total_weight == 0:
            match_score = 0.0
        else:
            weighted_scores = [
                count_matches["player"] * reference_counts["player"],
                count_matches["goalkeeper"] * reference_counts["goalkeeper"],
                count_matches["referee"] * reference_counts["referee"],
                count_matches["ball"] * reference_counts.get("soccer ball", 0)
            ]
            match_score = sum(weighted_scores) / total_weight
        
        logger.info(f"Count matches: {count_matches}")
        logger.info(f"Overall match score: {match_score:.3f}")
        
        return {
            "reference_counts": reference_counts,
            "high_confidence_counts": high_confidence_counts,
            "count_matches": count_matches,
            "match_score": match_score
        }

    async def validate_frame_detections(
        self,
        frame: np.ndarray,
        detections: dict,
        frame_idx: int
    ) -> dict:
        """Validate all detections in a frame concurrently."""
        # Get reference counts first
        reference_counts = self.get_reference_counts(frame)
        
        # Filter detections early
        filtered_detections = self.filter_detections(detections, frame.shape)
        
        validation_results = {
            "objects": [],
            "keypoints": {"score": 0.0, "points": [], "visualization_path": ""},
            "scores": {"keypoint_score": 0.0, "bbox_score": 0.0, "final_score": 0.0}
        }
        
        try:
            # Validate keypoints
            keypoints = filtered_detections.get("keypoints", [])
            keypoint_score = await self.validate_keypoints(frame, keypoints, frame_idx)
            validation_results["keypoints"].update({
                "score": keypoint_score,
                "points": keypoints,
                "visualization_path": f"frame_{frame_idx}_keypoints.jpg"
            })
            
            # Validate objects
            for i, obj in enumerate(filtered_detections.get("objects", [])):
                bbox = obj["bbox"]
                class_id = obj["class_id"]
                
                if class_id == BALL_CLASS_ID:
                    expected_class = "soccer ball"
                elif class_id == GOALKEEPER_CLASS_ID:
                    expected_class = "goalkeeper"
                elif class_id == REFEREE_CLASS_ID:
                    expected_class = "referee"
                else:  # PLAYER_CLASS_ID
                    expected_class = "soccer player"
                
                cropped = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                result = self.validate_bbox_content(cropped, expected_class)
                
                validation_results["objects"].append({
                    "bbox_idx": obj["id"],
                    "class": expected_class,
                    "class_id": obj["class_id"],
                    "probability": result
                })
            
            # Calculate scores
            bbox_score = self.calculate_bbox_confidence_score(validation_results)
            
            # Compare with reference counts
            count_validation = self.compare_with_reference_counts(reference_counts, validation_results)
            
            # Update final score to include count matching
            final_score = self.calculate_final_score(
                keypoint_score=keypoint_score,
                bbox_score=bbox_score,
                count_match_score=count_validation["match_score"]
            )
            
            validation_results["scores"].update({
                "keypoint_score": keypoint_score,
                "bbox_score": bbox_score,
                "count_match_score": count_validation["match_score"],
                "final_score": final_score
            })
            
            # Add reference count validation results
            validation_results["count_validation"] = count_validation
            
            logger.info(f"Frame {frame_idx} validation complete. Final score: {final_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error in frame validation: {str(e)}")
            logger.exception("Full error traceback:")
        
        return validation_results

    def calculate_bbox_confidence_score(self, bbox_validations: dict) -> float:
        """
        Calculate confidence score based on number of high-confidence detections.
        Returns score between 0 and 1.
        """
        high_confidence = 0
        total = 0
        
        for obj in bbox_validations.get("objects", []):
            total += 1
            if obj["probability"] >= 0.9:
                high_confidence += 1
        
        return high_confidence / total if total > 0 else 0.0

    def calculate_final_score(self, keypoint_score: float, bbox_score: float, count_match_score: float) -> float:
        """
        Calculate final score combining keypoint, bounding box, and count match scores.
        Weights: Keypoints 30%, Bounding Boxes 40%, Count Matching 30%
        """
        return (0.3 * keypoint_score) + (0.4 * bbox_score) + (0.3 * count_match_score)

    async def evaluate_response(
        self, 
        response: GSRResponse, 
        challenge: GSRChallenge,
        video_path: Path
    ) -> ValidationResult:
        """Evaluate a GSR response"""
        try:
            if not hasattr(response, 'response_id') or response.response_id is None:
                logger.error("Response object missing response_id")
                raise ValueError("Response object must have response_id set")
                
            # Select random frames to evaluate
            chosen_frames = self.select_random_frames(video_path)
            logger.info(f"Selected {len(chosen_frames)} frames for validation: {chosen_frames}")
            
            # Create debug frames directory
            debug_dir = Path("debug_frames")
            debug_dir.mkdir(exist_ok=True)
            
            # Evaluate all frames
            frame_evaluations = []
            for frame_num in chosen_frames:
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    logger.error(f"Failed to read frame {frame_num}")
                    continue
                
                frame_data = response.frames.get(str(frame_num), {})
                evaluation = await self.validate_frame_detections(frame, frame_data, frame_num)
                
                # Save debug frames
                annotated_frame = self.draw_annotations(frame, frame_data)
                debug_path = debug_dir / f"frame_{frame_num}_annotated.jpg"
                cv2.imwrite(str(debug_path), annotated_frame)
                
                evaluation['frame_number'] = frame_num
                frame_evaluations.append(evaluation)
            
            if not frame_evaluations:
                logger.error("No frames were successfully evaluated")
                return ValidationResult(
                    score=0.0,
                    frame_scores={},
                    feedback="Failed to evaluate any frames successfully"
                )
            
            # Process evaluations
            scores = []
            frame_scores = {}
            
            for eval_data in frame_evaluations:
                frame_num = eval_data["frame_number"]
                score = eval_data["scores"]["final_score"]
                
                # Store frame evaluation in database
                if hasattr(self, 'db_manager'):
                    self.db_manager.store_frame_evaluation(
                        response_id=response.response_id,
                        challenge_id=challenge.challenge_id,
                        miner_hotkey=response.miner_hotkey,
                        node_id=response.node_id,
                        frame_id=frame_num,
                        frame_timestamp=frame_num / 30.0,  # Assuming 30fps
                        frame_score=score,
                        raw_frame_path=str(debug_dir / f"frame_{frame_num}_raw.jpg"),
                        annotated_frame_path=str(debug_dir / f"frame_{frame_num}_annotated.jpg"),
                        vlm_response=eval_data["count_validation"]["reference_counts"],
                        feedback=""  # Keeping feedback empty as requested
                    )
                
                scores.append(score)
                frame_scores[frame_num] = score
            
            # Calculate final score
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            logger.info(f"Validation complete. Average score: {avg_score:.3f}")
            return ValidationResult(
                score=avg_score,
                frame_scores=frame_scores,
                feedback=""  # Keeping feedback empty as requested
            )
            
        except Exception as e:
            error_msg = f"Error evaluating response {response.challenge_id}: {str(e)}"
            logger.error(error_msg)
            return ValidationResult(
                score=0.0,
                frame_scores={},
                feedback=error_msg
            )

    async def download_video(self, video_url: str) -> Path:
        """Download video to temporary file, handling redirects and Google Drive URLs"""
        # Check cache first
        if video_url in self._video_cache:
            cached_path = self._video_cache[video_url]
            if cached_path.exists():
                logger.info(f"Using cached video from: {cached_path}")
                return cached_path
            else:
                # Remove from cache if file no longer exists
                del self._video_cache[video_url]

        logger.info(f"Downloading video from: {video_url}")
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Handle Google Drive URLs
            if 'drive.google.com' in video_url:
                logger.debug("Detected Google Drive URL, extracting file ID...")
                # Extract file ID
                file_id = None
                if 'id=' in video_url:
                    file_id = video_url.split('id=')[1].split('&')[0]
                elif '/d/' in video_url:
                    file_id = video_url.split('/d/')[1].split('/')[0]
                
                if not file_id:
                    raise ValueError("Could not extract Google Drive file ID from URL")
                
                # Use the direct download URL
                video_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
                logger.debug(f"Using direct download URL: {video_url}")

            try:
                logger.debug("Starting video download...")
                response = await client.get(video_url)
                response.raise_for_status()

                # Save to temporary file
                temp_dir = Path(tempfile.gettempdir())
                video_path = temp_dir / f"video_{datetime.now().timestamp()}.mp4"
                video_path.write_bytes(response.content)
                logger.info(f"Video downloaded to: {video_path}")
                
                # Verify the downloaded file
                if not video_path.exists() or video_path.stat().st_size == 0:
                    raise ValueError("Downloaded video file is empty or does not exist")
                
                # Try opening with OpenCV to verify it's a valid video
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    cap.release()
                    video_path.unlink(missing_ok=True)
                    raise ValueError("Downloaded file is not a valid video")
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                logger.info(f"Valid video file: {frame_count} frames, {fps} FPS")
                
                # Cache the video path
                self._video_cache[video_url] = video_path
                
                return video_path
                
            except httpx.HTTPError as e:
                raise ValueError(f"Failed to download video: {str(e)}")
            except Exception as e:
                raise ValueError(f"Error downloading video: {str(e)}")

    def extract_frame(self, video_path: Path, frame_num: int) -> np.ndarray:
        """Extract frame from video at given frame number"""
        logger.debug(f"Extracting frame {frame_num} from {video_path}")
        cap = cv2.VideoCapture(str(video_path))

        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # 0-based index
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not extract frame {frame_num}")
        logger.debug(f"Successfully extracted frame {frame_num}")
        return frame

    def encode_image(self, image: np.ndarray) -> str:
        """Encode image as base64 string"""
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).decode('utf-8')

    async def validate_response(
        self, 
        response: GSRResponse, 
        challenge: GSRChallenge,
        video_path: Path
    ) -> ValidationResult:
        """Validate GSR response against challenge"""
        return await self.evaluate_response(response, challenge, video_path)
