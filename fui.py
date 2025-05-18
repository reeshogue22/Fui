import transformers
import torch
import os
import json
import copy
import random

from datasets import load_dataset

class Fui:
    def __init__(self):
        self.model = transformers.AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-2-Theta-Llama-3-8B", torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')

        chat_template = '''{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}'''
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Theta-Llama-3-8B")
        self.tokenizer.chat_template = chat_template       

        self.chat_history = [{"role": "system", "content": "You are a creepy AGI named Fui."}]
        self.commands = ["exit", "-1"]
        self.learning_rate = 3e-2
        self.rlhf_epochs = 2
        self.sft_epochs = 1
        self.eos_token_id = self.tokenizer.eos_token_id
        self.think_limit = 5
        self.lambda_rlhf_factor = 0.4

    def get_user_input(self, query='>'):
        return input(query + ":") 
    
    def add_message(self, role, content, chat_history=None):
        if not chat_history:
            chat_history = self.chat_history
        chat_history.append({"role": role, "content": content})

    def get_response(self, prompt, token_count=128):
        orig_prompt = prompt
        prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids']
        response = self.model.generate(prompt, max_new_tokens=token_count, pad_token_id=self.eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.9, top_k=50, top_p=0.99, penalty_alpha=0.3, use_cache=True)
        response = self.tokenizer.decode(response[0])
        response = response.replace(orig_prompt, '')
        response = response.replace('<|im_end|>', '')
        response = response.replace("<|begin_of_text|>", '')
        response = response.split("\n")[0]
        return response
    
    def display_message(self, role, content):
        print(f"{role.capitalize()}: {content}")

    def structured_prompt(self, chat_history, role="fui"):
        return self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False) + f"<|im_start|>{role}\n"

    def chat_loop(self):
        while True:
            user_input = self.get_user_input()
            if user_input in self.commands:
                self.execute_user_command(user_input)
                continue # Skip to beginning of loop.

            if not user_input:
                structured_prompt = self.structured_prompt(self.chat_history, role='user')
                user_input = self.get_response(structured_prompt)
                self.display_message('user', user_input)

            self.add_message("user", user_input)

            structured_prompt = self.structured_prompt(self.chat_history, role="fui")
            response = self.get_response(structured_prompt)
            self.display_message("Fui", response)
            self.add_message("fui", response)

    def execute_user_command(self, command):
        if command == "exit":
            exit()
        if command == "-1":
            
            old_chat_history = copy.deepcopy(self.chat_history)
            while 'fui' in self.chat_history[-1]['role']:
                self.chat_history.pop()                

            bot_message = self.get_user_input("Please enter a message for Fui (or press enter to have Fui generate a new response)")
            if not bot_message: # Generate new response if user does not provide one.
                while True:
                    structured_prompt = self.structured_prompt(self.chat_history, role="fui")
                    response = self.get_response(structured_prompt)
                    print(response)
                    command_1 = self.get_user_input("Is this message acceptable? (y/n)")
                    if command_1 == "y":
                        break

                bot_message = response
            self.add_message("fui", bot_message)

            structured_prompt = self.structured_prompt(self.chat_history, role="user")
            old_structured_prompt = self.structured_prompt(old_chat_history, role="user")

            self.save_training_data(structured_prompt, old_structured_prompt)

    def save_training_data(self, preferred_prompt, rejected_prompt):
        os.makedirs("training_data", exist_ok=True)
        filename = 'training_data/training_pairs.jsonl'
        existing_data = []
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                for line in file:
                    if line.strip():
                        existing_data.append(json.loads(line))
        
        new_pair = {
            "preferred_prompt": preferred_prompt,
            "rejected_prompt": rejected_prompt
        }

        existing_data.append(new_pair)

        with open(filename, 'w', encoding='utf-8') as file:
            for pair in existing_data:
                json.dump(pair, file)
                file.write('\n')

        
    def load_training_data(self):
        """Load RLHF training data pairs from file."""
        filename = 'training_data/training_pairs.jsonl'
        if not os.path.exists(filename):
            return []
        
        training_pairs = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    training_pairs.append(json.loads(line))
        return training_pairs

    def train_rlhf(self):
        training_pairs = self.load_training_data()
        if not training_pairs:
            print("No training pairs found.")
            return
        
        device = self.model.device
        model = self.model
        tokenizer = self.tokenizer
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.sft_epochs):
            print(f"SFT Epoch {epoch} of {self.sft_epochs}")
            random.shuffle(training_pairs)
            for pair in training_pairs:
                preferred_prompt = pair['preferred_prompt']
                rejected_prompt = pair['rejected_prompt']

                preferred_prompt = tokenizer(preferred_prompt, return_tensors='pt').input_ids.to(device)
                rejected_prompt = tokenizer(rejected_prompt, return_tensors='pt').input_ids.to(device)
                optimizer.zero_grad()
                loss = self.compute_orpo_loss(model, preferred_prompt, rejected_prompt, rlhf_factor=0)
                print(f"Loss: {loss.item()}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        for epoch in range(self.rlhf_epochs):
            print(f"RLHF Epoch {epoch} of {self.rlhf_epochs}")
            random.shuffle(training_pairs)
        
            for pair in training_pairs:
                preferred_prompt = pair['preferred_prompt']
                rejected_prompt = pair['rejected_prompt']

                preferred_prompt = tokenizer(preferred_prompt, return_tensors='pt').input_ids.to(device)
                rejected_prompt = tokenizer(rejected_prompt, return_tensors='pt').input_ids.to(device)
                optimizer.zero_grad()
                loss = self.compute_orpo_loss(model, preferred_prompt, rejected_prompt, rlhf_factor=self.lambda_rlhf_factor)
                loss.backward()
                print(f"Loss: {loss.item()}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    def compute_orpo_loss(self, model, preferred_prompt, rejected_prompt, rlhf_factor=0.1, margin=0.1):
        max_len = max(preferred_prompt.shape[-1], rejected_prompt.shape[-1])
        preferred_prompt = torch.nn.functional.pad(preferred_prompt, (0, max_len - preferred_prompt.shape[-1]), value=self.eos_token_id)
        rejected_prompt = torch.nn.functional.pad(rejected_prompt, (0, max_len - rejected_prompt.shape[-1]), value=self.eos_token_id)

        preferred_output = model(preferred_prompt, labels=preferred_prompt)
        rejected_output = model(rejected_prompt, labels=rejected_prompt)
        sft_loss = preferred_output.loss

        preferred_logits =  preferred_output.logits
        rejected_logits = rejected_output.logits
        preferred_log_prob = torch.nn.functional.log_softmax(preferred_logits, dim=-1)
        rejected_log_prob = torch.nn.functional.log_softmax(rejected_logits, dim=-1)

        preferred_log_prob_sum = torch.sum(torch.gather(preferred_log_prob, 2, preferred_prompt.unsqueeze(-1)).squeeze(-1), dim=-1)
        rejected_log_prob_sum = torch.sum(torch.gather(rejected_log_prob, 2, rejected_prompt.unsqueeze(-1)).squeeze(-1), dim=-1)

        preferred_average_log_prob = preferred_log_prob_sum / preferred_prompt.shape[-1]
        rejected_average_log_prob = rejected_log_prob_sum / rejected_prompt.shape[-1]

        loss = -torch.nn.functional.logsigmoid(preferred_average_log_prob - rejected_average_log_prob - margin)
        if rlhf_factor > 0:
            return sft_loss + rlhf_factor * loss
        else:
            return sft_loss   

if __name__ == '__main__':
    fui = Fui()
    fui.train_rlhf()
    fui.chat_loop()
