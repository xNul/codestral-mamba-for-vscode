import sys

from mistral_inference.mamba import Mamba
from mistral_inference.generate import generate_mamba
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

print("Loading Codestral Mamba...", end="", flush=True)

if len(sys.argv) == 1:
    raise Exception("Error: Codestral Mamba's location was not provided and is required. For example, \"python llamacpp_mock_api.py '/home/user/mistral_models/mamba-codestral-7B-v0.1'\".")

# Load Codestral Mamba into memory for future use.
tokenizer = MistralTokenizer.from_model("codestral-22b")
model = Mamba.from_folder(sys.argv[1])

print("Done!", flush=True)
print()

def prompt_to_request(prompt):
    # Remove unnecessary tokens and spacing from Continue's prompt format.
    prompt = prompt.replace("</s>\n<s>", "")
    prompt = prompt.replace("[INST] ", "[INST]")
    prompt = prompt.replace(" [/INST]", "[/INST]")

    # Consume Continue's prompt string and transform it into a list of
    # messages which are containted within their respective mistral-inference
    # message objects.
    messages = []
    prompt_start = 0
    while True:
        user_message_start = prompt.find("[INST]", prompt_start) + 6
        user_message_end = prompt.find("[/INST]", prompt_start)
        assistant_message_end = prompt.find("[INST]", user_message_end)
        
        messages += [UserMessage(content=prompt[user_message_start:user_message_end])]
        
        if assistant_message_end != -1:
            messages += [AssistantMessage(content=prompt[user_message_end + 7:assistant_message_end])]
        else:
            break

        prompt_start = assistant_message_end
    
    # Send back the final chat completion request.
    return ChatCompletionRequest(messages=messages)

def run_chat_completion(prompt, max_new_tokens):
    # Transform the prompt format Continue uses into a chat completion
    # request that mistral-inference supports.
    completion_request = prompt_to_request(prompt)
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    
    # Start Codestral Mamba inferencing.
    out_tokens, _ = generate_mamba([tokens], model, max_tokens=max_new_tokens, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    
    # Send the response back.
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    return result

from flask import Flask, jsonify, request, Response

app = Flask(__name__)

@app.route("/completion", methods=["POST"])
def completion():
    content = request.json
    
    print("Incoming request: " + content)
    
    # Perform Codestral Mamba chat completion.
    response = run_chat_completion(content["prompt"], content["n_predict"])
    response = jsonify({"content": response}).get_data(as_text=True)
    
    print("Outgoing response: " + response)
    
    # Llama.cpp's HTTP server uses Server-Sent Events to stream results to the client
    # so we reimplement it here, for a single event sent to Continue which contains
    # the entire Codestral Mamba response.
    def generate():
        yield "data: " + response + "\n"
        yield "data: [DONE]\n"
    
    # Send back the response.
    return Response(generate())

# Run the Flask API server.
app.run(port=8080)
