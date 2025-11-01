import os
import json
import sys
import boto3
import botocore.exceptions

prompt = "you are a smart assistant so let me know what is a machine learning"

# ensure region is set or override here
region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"

try:
    bedrock = boto3.client("bedrock-runtime", region_name=region)
except Exception as e:
    print("Failed to create boto3 client:", e)
    sys.exit(1)

payload = {
    "prompt": f"[INST] {prompt} [/INST]",
    "max_gen_len": 512,
    "temperature": 0.3,
    "top_p": 0.9,
}

body = json.dumps(payload)
model_id = "meta.llama3-70b-instruct-v1:0"

try:
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )
except botocore.exceptions.NoCredentialsError:
    print("No AWS credentials found. Configure AWS credentials or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY.")
    sys.exit(1)
except Exception as e:
    print("invoke_model failed:", e)
    sys.exit(1)

print("Raw response keys:", list(response.keys()))

# response['body'] can be a StreamingBody or bytes/str depending on SDK; handle all cases
raw_body = response.get("body")
decoded = None
try:
    if hasattr(raw_body, "read"):
        decoded = raw_body.read().decode("utf-8")
    elif isinstance(raw_body, (bytes, bytearray)):
        decoded = raw_body.decode("utf-8")
    else:
        decoded = str(raw_body)
except Exception as e:
    print("Failed to read/decode response body:", e)
    decoded = str(raw_body)

print("Decoded body:", decoded)

# try parse JSON, fallback to raw text
try:
    response_body = json.loads(decoded)
except Exception:
    response_body = decoded

print("Parsed response_body:", response_body)

if isinstance(response_body, dict):
    if "generation" in response_body:
        result = response_body["generation"]
    elif "outputs" in response_body and response_body["outputs"]:
        result = response_body["outputs"][0].get("text", "")
    elif "output_text" in response_body:
        result = response_body["output_text"]
    else:
        result = response_body
else:
    result = response_body

print("Result:\n", result)