# GPTQ-finetune
train a GPTQ quantised LLM with a local or hosted custon dataset using PEFT (parameter efficient fine tuning)  and/or run inference with/without the model adapter, this useful for multiple fine-tuned agents

the generated adapter is here [sujitvasanth/TheBloke-openchat-3.5-0106-GPTQ-PEFTadapterJsonSear ](https://huggingface.co/sujitvasanth/TheBloke-openchat-3.5-0106-GPTQ-PEFTadapterJsonSear)

training dataset used: https://huggingface.co/datasets/sujitvasanth/jsonsearch2 there id s csv version in this repository

# Intructions
run GPTQ-finetune.py adjusting the location of your GPTQ model, csv file and where to save the generated model adapter
the example dataset improves the json search capabilities of The Bloke's openchat GPTQ

![Screenshot from 2022-06-29 03-47-08-crop](https://github.com/sujitvasanth/GPTQlocalcustomPEFTdualinference/assets/18464444/72223b31-88e3-4c8d-953a-7d433b18449d)
![Screenshot from 2024-02-26 04-26-10](https://github.com/sujitvasanth/GPTQlocalcustomPEFTdualinference/assets/18464444/8c7c2237-2891-4754-b929-e343c65f55b3)
