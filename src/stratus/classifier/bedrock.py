from __future__ import annotations
import base64
import cv2
import json

from stratus.core.vision import CameraFrame
from stratus.core.arm_driver import TriageCommand

try:
    import boto3
except ImportError:
    boto3 = None


class BedrockClassifier:
    """Uses Amazon Bedrock Converse API to grade server components.
    Requires AWS credentials configured (~/.aws/credentials or env vars).
    """

    def __init__(self, model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"):
        if boto3 is None:
            raise ImportError("boto3 not installed. Run: pip install boto3")
        self._client = boto3.client("bedrock-runtime")
        self._model_id = model_id

    def set_background(self, frame: CameraFrame) -> None:
        pass

    def classify(self, frame: CameraFrame) -> TriageCommand:
        _, buffer = cv2.imencode(".jpg", frame.image)
        response = self._client.converse(
            modelId=self._model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"text": (
                        "You are an ITAD component inspector. "
                        "Examine this server component image. "
                        "Respond with ONLY a JSON object: "
                        '{"grade": "A"|"B"|"C", "reason": "short reason"}'
                        "A = refurbishable, B = needs repair, C = scrap/recycle"
                    )},
                    {"image": {"format": "jpeg", "source": {"bytes": buffer.tobytes()}}},
                ],
            }],
            inferenceConfig={"maxTokens": 200, "temperature": 0},
        )

        text = response["output"]["message"]["content"][0]["text"]
        result = json.loads(text)
        grade = result["grade"].upper()
        bin_map = {"A": "bin_a", "B": "bin_b", "C": "bin_c"}
        target = bin_map.get(grade, "bin_c")

        return TriageCommand(
            action="pick_and_place",
            target_bin=target,
            pickup_pose={"x": 0.25, "y": 0.0, "z": 0.15, "roll": 0, "pitch": 0.4, "yaw": 0},
            drop_pose={"x": 0.15, "y": -0.2, "z": 0.15, "roll": 0, "pitch": 0.4, "yaw": 0},
        )
