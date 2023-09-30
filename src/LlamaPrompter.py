from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "TheBloke/Llama-2-7B-GPTQ"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

classes_path = "data/CUB_200_2011/CUB_200_2011/classes.txt"

classes = []

with open(classes_path) as f:
    classes_lines = f.readlines()

for line in classes_lines:
    classes.append(line.split(".")[1].strip().replace("_", " ").lower())

EPOCHS = 10
for i in range(EPOCHS):
    for class_name in classes:
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
Q: What are useful visual features to distinguish {class_name} in a photo?
A: There are several useful visual feature to tell there is a {class_name} in a photo:\n"""

        # generate a response (takes several seconds)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
        output = tokenizer.decode(output[0])

        buffer = ""

        lines = output.split("\n")
        for line in lines:
            if line.startswith("Q:"):
                break
            buffer += line + "\n"
            
        # display the response
        with open(f"results/CUB-Attributes/{class_name}.txt", "a") as text_file:
            text_file.write(buffer)