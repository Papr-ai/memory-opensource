import os
import io
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from dotenv import find_dotenv, load_dotenv
import grpc
from services.logging_config import get_logger

# Create a logger instance for this module
logger = get_logger(__name__)  # Will use 'api_handlers.stable_diffusion' as the logger name



ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

    

class StableDiffusion:
    def __init__(self):
        logger.info("Initializing Stability API client...")
        self.stability_api = client.StabilityInference(
            key=os.environ['STABILITY_KEY'],
            verbose=True,
            engine="stable-diffusion-xl-1024-v1-0"
        )
        logger.info("Stability API client initialized successfully.")

    def generate_image(self, text, seed=None):
        logger.info(f"Generating image for prompt: {text}")
        #self._request_timeout = 60.0

        answers = self.stability_api.generate(
            prompt=text,
            seed=seed,
            steps=50,
            cfg_scale=8.0,
            width=1024,
            height=1024,
            samples=1,
            sampler=generation.SAMPLER_K_DPMPP_2M
        )

        try:
            
            # process the response
            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        warning_message = "Your request activated the API's safety filters and could not be processed. Please modify the prompt and try again."
                        logger.warning(warning_message)
                        warnings.warn(warning_message)
                        return None
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        logger.info("Image generated successfully.")
                        img = Image.open(io.BytesIO(artifact.binary))
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        return img_byte_arr.getvalue()

        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.details()}")
            # Handle the error or retry the request

