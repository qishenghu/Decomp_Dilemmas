"""
File: claim_verifier.py

Description:
This file is an modified version based on the amazing VeriScore repository.
VeriScore Repo: https://github.com/Yixiao-Song/VeriScore
"""

import asyncio
import json
import os
import pdb
from math import exp

import requests
from get_response import GetResponse
from minicheck.minicheck import MiniCheck


class ClaimVerifier():
    def __init__(self, model_name, label_n=2, cache_dir="./data/cache/", demon_dir="data/demos/",
                 use_external_model=False, use_nli_verification=False):
        self.model = None
        self.model_name = model_name
        self.label_n = label_n
        self.use_nli_verification = use_nli_verification
        self.system_message = None
        self.evaluator = None
        if 'minicheck' in self.model_name.lower() and self.use_nli_verification:
            if self.model_name == 'minicheck_flan_t5':
                self.model = MiniCheck(model_name='flan-t5-large', enable_prefix_caching=False)
            else:
                # By default, use the Bespoke-MiniCheck-7B model
                self.model = MiniCheck(model_name='Bespoke-MiniCheck-7B', enable_prefix_caching=False)

        if self.use_nli_verification:
            if 'alignscore' in model_name.lower():
                print(f"Using alignscore for NLI verification")
            elif 'minicheck' in model_name.lower():
                print(f"Using minicheck for NLI verification")
            else:
                raise ValueError(f"Unknown model name: {model_name} for NLI verification")
        
        if os.path.isdir(model_name) and use_external_model:
            from unsloth import FastLanguageModel

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            self.tokenizer.padding_side = "left"
            self.alpaca_prompt = open("./prompt/verification_alpaca_template.txt", "r").read()
            self.instruction = open("./prompt/verification_instruction_binary_no_demo.txt", "r").read()
        elif not use_nli_verification:
            cache_dir = os.path.join(cache_dir, model_name)
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, "claim_verification_cache.json")
            self.demon_path = os.path.join(demon_dir, 'few_shot_examples.jsonl')
            self.get_model_response = GetResponse(cache_file=self.cache_file,
                                                  model_name=model_name,
                                                  max_tokens=1000,
                                                  temperature=0,
                                                  save_interval=2000)
            self.system_message = "You are a helpful assistant who can verify the truthfulness of a claim against reliable external world knowledge."
            self.prompt_initial_temp = self.get_initial_prompt_template()
        self.lock = asyncio.Lock()  # 用于避免写入冲突
        self.semaphore = asyncio.Semaphore(10)  # 控制并发量
    
    

    def get_alignscore(self, evidence="", claim=""):
        # Define the server URL
        url = "http://0.0.0.0:5000/alignscore_large"
        
        # Define the data to send
        data = {
            'evidence': evidence,
            'claim': claim,
            'type': 'nli_sp'
        }
        
        try:
            # Send a POST request
            response = requests.post(url, json=data)
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Extract and return the alignscore from the response
            return response.json().get('alignscore')
            
        except requests.exceptions.RequestException as e:
            print(f"!!!!!An error occurred: {e}")
            return None


    def get_instruction_template(self):
        prompt_temp = ''
        if self.label_n == 2:
            prompt_temp = open("./prompt/verification_instruction_binary_nli.txt", "r").read()
        elif self.label_n == 3:
            prompt_temp = open("./prompt/verification_instruction_trinary.txt", "r").read()
        else:
            raise ValueError(f"Label number {self.label_n} is not supported.")
        return prompt_temp

    def get_initial_prompt_template(self):
        prompt_temp = self.get_instruction_template()
        with open(self.demon_path, "r") as f:
            example_data = [json.loads(line) for line in f if line.strip()]
        element_lst = []
        for dict_item in example_data:
            claim = dict_item["claim"]
            search_result_str = dict_item["search_result"]
            human_label = dict_item["human_label"]
            if self.label_n == 2:
                if human_label.lower() in ["support.", "supported.", "supported", "support"]:
                    human_label = "Supported."
                else:
                    human_label = "Unsupported."
            element_lst.extend([claim, search_result_str, human_label])

        prompt_few_shot = prompt_temp.format(*element_lst)

        self.your_task = "Your task:\n\n{search_results}\n\nClaim: {claim}\n\nTask: Given the search results above, is the claim supported or unsupported? If supported, return 'Supported'. If not, return 'Unsupported'. For your decision, you must only output the word 'Supported', or the word 'Unsupported', nothing else.\n\nYour decision:"

        if self.label_n == 2:
            self.system_message = "You are a helpful assistant who can judge whether a claim is supported by the search results or not."
        elif self.label_n == 3:
            self.system_message = "You are a helpful assistant who can judge whether a claim is supported or contradicted by the search results, or whether there is no enough information to make a judgement."

        return prompt_few_shot

    def verifying_claim(self, claim_snippets_dict, search_res_num=5, cost_estimate_only=False):
        """
        search_snippet_lst = [{"title": title, "snippet": snippet, "link": link}, ...]
        """
        prompt_tok_cnt, response_tok_cnt = 0, 0
        out_lst = []
        claim_verify_res_dict = {}
        for claim, search_snippet_lst in claim_snippets_dict.items():
            search_res_str = ""
            search_cnt = 1

            claim_nli_score = None
            claim_nli_score_list = []
            for search_dict in search_snippet_lst[:search_res_num]:
                if 'title' and 'link' in search_dict:
                    search_res_str += f'Search result {search_cnt}\nTitle: {search_dict["title"].strip()}\nLink: {search_dict["link"].strip()}\nContent: {search_dict["snippet"].strip()}\n\n'
                else:
                    search_res_str += f'Search result {search_cnt}\nContent: {search_dict["snippet"].strip()}\n\n'
                search_cnt += 1
            
            if self.use_nli_verification:
                for search_dict in search_snippet_lst:
                    if 'alignscore' in self.model_name.lower():
                        nli_func = self.get_alignscore
                    else:
                        raise ValueError(f"Unknown model name: {self.model_name} for NLI verification")

                    claim_nli_score_list.append(nli_func(evidence=search_dict['snippet'].strip(), claim=claim))
                # use the max value
                claim_nli_score = max(claim_nli_score_list)
                clean_output=None
                prompt=None
                response=None
            else:
                if self.model:
                    usr_input = f"Claim: {claim.strip()}\n\n{search_res_str.strip()}"
                    formatted_input = self.alpaca_prompt.format(self.instruction, usr_input)
                    prompt=formatted_input
                    inputs = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")
                    output = self.model.generate(**inputs,
                                                max_new_tokens=500,
                                                use_cache=True,
                                                eos_token_id=[self.tokenizer.eos_token_id,
                                                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                                                pad_token_id=self.tokenizer.eos_token_id, )
                    response = self.tokenizer.batch_decode(output)
                    clean_output = ' '.join(response).split("<|end_header_id|>\n\n")[
                        -1].replace("<|eot_id|>", "").strip()
                else:
                    prompt_tail = self.your_task.format(
                        claim=claim,
                        search_results=search_res_str.strip(),
                    )
                    prompt = f"{self.prompt_initial_temp}\n\n{prompt_tail}"
                    response, prompt_tok_num, response_tok_num = self.get_model_response.get_response(self.system_message,
                                                                                                    prompt,
                                                                                                    cost_estimate_only=cost_estimate_only)
                    prompt_tok_cnt += prompt_tok_num
                    response_tok_cnt += response_tok_num

                    clean_output = response.replace("#", "").split(".")[0].lower() if response is not None else None
                
            
            claim_verify_res_dict = {"claim": claim,
                                     "search_results": search_res_str,
                                     "verification_result": clean_output, 
                                     "system_message": self.system_message,
                                     "prompt": prompt,
                                     "inital_response": response,
                                     "use_nli_verification": self.use_nli_verification,
                                     "claim_nli_score": claim_nli_score,
                                     "claim_nli_score_list": claim_nli_score_list,
                                     }
            out_lst.append(claim_verify_res_dict)
        return out_lst, prompt_tok_cnt, response_tok_cnt

    
    async def async_verifying_claim(self, claim_snippets_dict, search_res_num=5, cost_estimate_only=False):
        prompt_tok_cnt, response_tok_cnt = 0, 0
        out_lst = []
        tasks = []
        for claim, search_snippet_lst in claim_snippets_dict.items():
            tasks.append(self._verify_single_claim(claim, search_snippet_lst, search_res_num, cost_estimate_only))
        
        results = await asyncio.gather(*tasks)
        for result in results:
            out_lst.append(result[0])
            prompt_tok_cnt += result[1]
            response_tok_cnt += result[2]
        
        return out_lst, prompt_tok_cnt, response_tok_cnt
    


    async def _verify_single_claim(self, claim, search_snippet_lst, search_res_num, cost_estimate_only):
        async with self.semaphore:  # 使用信号量控制并发量
            search_res_str = ""
            search_cnt = 1
            claim_nli_score_list, claim_nli_score = [], None
            prompt_tok_cnt, response_tok_cnt = 0, 0
            for search_dict in search_snippet_lst[:search_res_num]:
                if 'title' and 'link' in search_dict:
                    search_res_str += f'Search result {search_cnt}\nTitle: {search_dict["title"].strip()}\nLink: {search_dict["link"].strip()}\nContent: {search_dict["snippet"].strip()}\n\n'
                else:
                    search_res_str += f'Search result {search_cnt}\nContent: {search_dict["snippet"].strip()}\n\n'
                search_cnt += 1
            
            if self.use_nli_verification:
                for search_dict in search_snippet_lst:
                    evi_context = search_dict['snippet'].strip()
                    if 'title' and 'link' in search_dict:
                        evi_context = f'Title: {search_dict["title"].strip()}\nLink: {search_dict["link"].strip()}\nContent: {search_dict["snippet"].strip()}\n\n'
                    
                    # claim_nli_score_list.append(self.get_alignscore(evidence=evi_context, claim=claim))
                    if 'alignscore' in self.model_name.lower():
                        claim_nli_score_list.append(self.get_alignscore(evidence=evi_context, claim=claim))
                    elif 'minicheck' in self.model_name.lower():
                        pred_label, entailment_scores, _, _ = self.model.score(docs=[evi_context], claims=[claim])
                        claim_nli_score_list.append(entailment_scores[0])
                    else:
                        raise ValueError(f"Unknown model: {self.model_name}")
                    
                    await asyncio.sleep(0.1)
                # use the max value
                if len(claim_nli_score_list) > 0:
                    claim_nli_score = max(claim_nli_score_list)
                else:
                    claim_nli_score = 0.
                clean_output=None
                prompt=None
                response=None
            else:
                # if self.model:
                #     usr_input = f"Claim: {claim.strip()}\n\n{search_res_str.strip()}"
                #     formatted_input = self.alpaca_prompt.format(self.instruction, usr_input)

                #     inputs = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")
                #     output = self.model.generate(**inputs,
                #                                 max_new_tokens=500,
                #                                 use_cache=True,
                #                                 eos_token_id=[self.tokenizer.eos_token_id,
                #                                             self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                #                                 pad_token_id=self.tokenizer.eos_token_id, )
                #     response = self.tokenizer.batch_decode(output)
                #     clean_output = ' '.join(response).split("<|end_header_id|>\n\n")[
                #         -1].replace("<|eot_id|>", "").strip()
                # else:
                prompt_tail = self.your_task.format(
                    claim=claim,
                    search_results=search_res_str.strip(),
                )
                prompt = f"{self.prompt_initial_temp}\n\n{prompt_tail}"
                response, prompt_tok_num, response_tok_num, response_logprobs = await self.get_model_response.async_get_response(self.system_message,
                                                                                                            prompt,
                                                                                                            cost_estimate_only=cost_estimate_only,
                                                                                                            logprobs=True)
                prompt_tok_cnt = prompt_tok_num
                response_tok_cnt = response_tok_num

                clean_output = response.replace("#", "").split(".")[0].lower() if response is not None else None
                if response_logprobs is not None:
                    joint_logprob = 0.
                    target_logprob = 0.
                    is_target_token = False
                    for token in response_logprobs:
                        token_str = token.token
                        token_logprob = token.logprob
                        if token_str.lower() in ['supported', 'unsupported']:
                            target_logprob = token_logprob
                            is_target_token = True
                            break
                        joint_logprob += token_logprob
                    if not is_target_token:
                        target_logprob = joint_logprob
                    prob = exp(target_logprob)
                    # this prob will be used for entailment score
                    if clean_output == 'supported':
                        claim_nli_score = prob
                    else:
                        claim_nli_score = 1 - prob
                    claim_nli_score_list.append(claim_nli_score)
                    print(f"response: {response}, claim_nli_score: {claim_nli_score}")
            
            claim_verify_res_dict = {"claim": claim,
                                     "search_results": search_res_str,
                                     "verification_result": clean_output, 
                                     "system_message": self.system_message,
                                     "prompt": prompt,
                                     "inital_response": response,
                                     "claim_nli_score": claim_nli_score,
                                     "claim_nli_score_list": claim_nli_score_list,
                                     }
            return claim_verify_res_dict, prompt_tok_cnt, response_tok_cnt

