import os 
import json
import sys
import boto3

prompts="""
    you are a smart assistant so let me know what is a machine learning 
    """
bedrock = boto3.client(service_name="bedrock-runtime")

payload={

}

body = json.dump(payload)
model_id = "meta.llama3-70b-instruct-v1:0"

response = bedrock.invoke_model(
    body=body, 
    model_id=model_id,
    accept="application/json"
)

response_body = json.loads(response.get("body").read())
response_text = response_body["generation"]
print(response_text)