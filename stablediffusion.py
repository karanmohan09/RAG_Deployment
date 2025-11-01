# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import boto3
from PIL import Image
from botocore.exceptions import ClientError


class ImageError(Exception):
    """Custom exception for errors returned by Amazon Titan Image Generator G1"""
    def __init__(self, message):
        self.message = message


def generate_image(model_id: str, body: str):
    """
    Generate an image using Amazon Titan Image Generator G1 model.
    Args:
        model_id (str): Model ID (e.g. 'amazon.titan-image-generator-v1')
        body (str): JSON request body
    Returns:
        image_bytes (bytes): Generated image data
    """
    bedrock = boto3.client(service_name="bedrock-runtime")

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    # Parse JSON response
    response_body = json.loads(response["body"].read())

    # Handle potential errors
    if "error" in response_body and response_body["error"]:
        raise ImageError(f"Image generation error: {response_body['error']}")

    # Extract and decode the base64 image
    base64_image = response_body["images"][0]
    image_bytes = base64.b64decode(base64_image)

    return image_bytes


def main():
    """Entrypoint for image generation"""

    model_id = "amazon.titan-image-generator-v1"

    # ✏️ Change this prompt to anything you want
    prompt = "A photograph of a Painting."

    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 1024,
            "width": 1024,
            "cfgScale": 8.0,
            "seed": 42
        }
    })

    try:
        image_bytes = generate_image(model_id, body)

        # Display the generated image
        image = Image.open(io.BytesIO(image_bytes))
        image.show()

        # Optionally, save it
        image.save("generated_image.png")

        print(f"✅ Image generated successfully and saved as 'generated_image.png'")

    except ClientError as e:
        print(f"❌ AWS Client Error: {e.response['Error']['Message']}")
    except ImageError as e:
        print(f"❌ Titan Image Generation Error: {e.message}")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")


if __name__ == "__main__":
    main()
