from transformers import (AutoTokenizer,AutoModelForCausalLM)
import torch

class LLMGenerator:

    def __init__(self,model_name: str):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        self.model.eval()


    def generate(self,prompt: str,max_new_tokens: int = 256) -> str:

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ] 
        '''这样做的意义是：
        'user' + prompt 变成<user>你喜欢吃什么？'''

        '''理解为在首尾封装<user>和<assistant>
        template：模版'''
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False, # 先变成<user>你喜欢吃什么？，先不编码
            add_generation_prompt=True # 在后面再加一个<assistant>
            # 意思是该AI生成了
        )

        # input_ids:（batch_size=1,seq_len)
        inputs = self.tokenizer(text,return_tensors="pt").to(self.model.device)

        with torch.no_grad():

            '''inputs = 
            {
                "input_ids": ...,
                "attention_mask": ...
            }'''

            '''可以在这里，后训练SFT时加入LoRA'''
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False #一般会趋向于每一步选概率最高的 token
            )

            # outputs(batch_size,seq_len+max_new_generate)

        outputs = outputs[:,inputs["input_ids"].shape[1]:]

        # 取第一个批次，因为只有一个query
        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True # 把<eos><assistant><pad>删了
        )

        return answer.strip()