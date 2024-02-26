from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, logging
from peft import LoraConfig, PeftModel, PeftConfig; from transformers_stream_generator import init_stream_support
import torch, warnings
model_id = "/home/sujit/Downloads/text-generation-webui-main/models/TheBloke_openchat-3.5-0106-GPTQ"
adapter_id = "/home/sujit/Downloads/text-generation-webui-main/models/sujitvasanth_openchat_adapter"
string = ["User: ","<|end_of_turn|>","Assistant: "]
warnings.filterwarnings("ignore", module="transformers")

if input("would you like to re-finetune the model? ").lower()=="y":
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorWithPadding
    from peft import prepare_model_for_kbit_training, get_peft_model
    from datasets import load_dataset;
    quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
    model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quantization_config_loading, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=8, lora_alpha=32, target_modules=["k_proj", "o_proj", "q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    if input("load the local version? ").lower()=="y":
        data = load_dataset("csv",data_files="/home/sujit/Downloads/jsonsearch3.csv")
    else:
        data = load_dataset("sujitvasanth/jsonsearch2")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokens = data.map(lambda examples: tokenizer([f"{string[0]}{user}{string[1]}\n{string[2]}{assistant}{string[1]}"
                                                      for user, assistant in zip(examples['User'], examples['Assistant'])],),
                          batched=True, remove_columns=["User", "Assistant"])
    for tokeens in tokens["train"]["input_ids"]: print(tokenizer.decode(tokeens, skip_special_tokens=False)+"\n")
    train_dataset, eval_dataset = tokens["train"].train_test_split(test_size=0.05,shuffle=True).values()

    training_args = TrainingArguments(
        output_dir="outputs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model,args=training_args,train_dataset=train_dataset,eval_dataset=eval_dataset,data_collator=data_collator,)
    trainer.train()
    model.save_pretrained(adapter_id)

logging.set_verbosity(logging.CRITICAL)
model = AutoModelForCausalLM.from_pretrained(model_id,
    quantization_config= GPTQConfig(bits=4, disable_exllama=False),device_map="auto") # is_trainable=True
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model.load_adapter(adapter_id)
#model = PeftModel.from_pretrained(model, adapter_id )
init_stream_support()
LORAcontext=Basecontext=""

def handle_prompt(context):
    prompt = context + string[0] + rawprompt + string[1] + "\n" + string[2]
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    generator = model.generate(input_ids, temperature=0.01, do_stream=True, do_sample=True, top_p=0.01, top_k=2, max_new_tokens=14000, stream=True)
    output = ""; last_tokens = []; firstline = False
    
    for index, x in enumerate(generator):
        tokens = x.cpu().tolist()
        word = tokenizer.decode(tokens)
        if "�" not in word:
            if firstline or (not firstline and word != "\n"):
                a = tokenizer.decode(last_tokens + tokens)
                if " " in a:
                    word = " " + word
                    last_tokens = []
                    firstline = True
                if "<" in a:
                    if ">" not in a:
                        last_tokens += tokens
                    else:
                        last_tokens = []
                    if string[1] in a:
                        context += string[0] + rawprompt + "\n" + string[2] + output + "\n"
                        print()
                        return output, context
                else:
                    last_tokens = tokens
                    output += word
                    print(word, end="")

model.enable_adapters()
while True:
    rawprompt=""
    while rawprompt=="":
        rawprompt=input(string[0])
        if rawprompt=="clear":
            LORAcontext=Basecontext=rawprompt=""
            print("context cleared")
    print(string[2]+"(LoRA)", end="")
    LoRAresponse, LORAcontext =handle_prompt(LORAcontext)
    print(string[2]+"(Basemodel)", end="")
    model.disable_adapters()
    Baseresponse, Basecontext=handle_prompt(Basecontext)
    if Baseresponse==LoRAresponse:
        print("resonses were the same")
    else:
        print("resonses DIFFERENT")
    model.enable_adapters()
