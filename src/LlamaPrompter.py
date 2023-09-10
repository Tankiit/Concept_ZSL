from llama_cpp import Llama
LLM = Llama(model_path="data/llama-2-7b.Q4_0.gguf")

class_name = "seagull"

# create a text prompt
prompt = f"""Q: What are useful visual features to distinguish a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet
Q: What are useful visual features to distinguish class name in a photo?
A: There are several useful visual feature to tell there is a {class_name} in a photo:\n"""

while True:
    # generate a response (takes several seconds)
    output = LLM.create_completion(prompt, max_tokens=512, top_k=100)["choices"][0]["text"]

    buffer = ""

    lines = output.split("\n")
    for line in lines:
        if line.startswith("Q:"):
            break
        buffer += line + "\n"
        
    # display the response
    with open("Output.txt", "a") as text_file:
        text_file.write(buffer)