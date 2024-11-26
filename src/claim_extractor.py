"""
File: claim_extractor.py

Description:
This file is an modified version based on the amazing VeriScore repository.
VeriScore Repo: https://github.com/Yixiao-Song/VeriScore
"""

import asyncio
import json
import os
import pdb
import re

import regex
import spacy
from get_response import GetResponse
from third_party.factscore.atomic_facts import AtomicFactGenerator
from third_party.self_diagnosis import (extract_detection, extract_diagnosis,
                                        get_detection_prompt,
                                        get_self_diagnosis_prompt)

from third_party.specified_number_claims import (
    CLAIM_LEVEL_DECOMPOSE_NUM_PROMPT_TEMPLATE,
    RESPONSE_LEVEL_DECOMPOSE_NUM_PROMPT_TEMPLATE)
from third_party.wice_decompose import WICE_PROMPT
from tqdm import tqdm


class ClaimExtractor():
    def __init__(self, model_name, cache_dir="./data/cache/", use_external_model=False, input_level="claim", model_name_diagnosis=None):
        self.model = None
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(2)
        self.input_level = input_level
        self.model_name_diagnosis = model_name_diagnosis
        self.specified_number_prompt_template = RESPONSE_LEVEL_DECOMPOSE_NUM_PROMPT_TEMPLATE
        if os.path.isdir(model_name) or use_external_model:
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            self.model = self.model.to("cuda")
            self.alpaca_prompt = open("./prompt/extraction_alpaca_template.txt", "r").read()
        else:
            cache_dir = os.path.join(cache_dir, model_name)
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, f"claim_extraction_cache.json")
            self.get_model_response = GetResponse(cache_file=self.cache_file,
                                                  model_name=model_name,
                                                  max_tokens=2048,
                                                  temperature=0)
            if self.model_name_diagnosis == model_name:
                self.get_diagnosis_response = self.get_model_response
            else:
                cache_file_diagnosis = os.path.join(cache_dir, self.model_name_diagnosis)
                self.get_diagnosis_response = GetResponse(cache_file=cache_file_diagnosis,
                                                  model_name=self.model_name_diagnosis,
                                                  max_tokens=2048,
                                                  temperature=0)
            
            self.system_message = "You are a helpful assistant who can extract verifiable atomic claims from a piece of text. Each atomic fact should be verifiable against reliable external world knowledge (e.g., via Google Search, Wikipedia, etc.)"

        self.spacy_nlp = spacy.load('en_core_web_sm')

    def non_qa_scanner_extractor(self, response, cost_estimate_only=False):
        """
        Given a model output
        - split the response into sentences using spaCy
        - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """
        sentences = self.get_sentence(response)

        all_facts_lst = []
        # keep track of token counts
        prompt_tok_cnt, response_tok_cnt = 0, 0

        # new return values
        snippet_lst = []
        fact_lst_lst = []

        for i, sentence in enumerate(sentences):
            if self.model:
                input = response.strip()
                snippet = input.replace(sentence, f"<SOS>{sentence}<EOS>")
            else:
                lead_sent = sentences[0]  # 1st sentence of the para
                context1 = " ".join(sentences[max(0, i - 3):i])
                sentence = f"<SOS>{sentences[i].strip()}<EOS>"
                context2 = " ".join(sentences[i + 1:i + 2])

                # if the para is not long
                if len(sentences) <= 5:
                    snippet = f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
                # if the para is long, add lead sentence to context1
                else:
                    snippet = f"{lead_sent.strip()} {context1.strip()} {sentence.strip()} {context2.strip()}".strip()

            snippet_lst.append(snippet)

            # call fact_extractor on each snippet
            facts, prompt_tok_num, response_tok_num = self.fact_extractor(snippet, sentences[i].strip(),
                                                                          qa_input=False,
                                                                          cost_estimate_only=cost_estimate_only)

            # update token counts
            prompt_tok_cnt += prompt_tok_num
            response_tok_cnt += response_tok_num

            if facts == None:
                fact_lst_lst.append([None])
                continue

            # deduplication
            fact_lst = []
            for fact in facts:
                if fact.strip() == "":
                    continue
                # cases where GPT returns its justification
                elif fact.startswith("Note:"):
                    continue
                elif fact not in all_facts_lst:
                    all_facts_lst.append(fact.strip())
                fact_lst.append(fact.strip())
            fact_lst_lst.append(fact_lst)
        
        if len(all_facts_lst) == 0:
            # If no facts are extracted, just use the original response as the only fact
            all_facts_lst = [response.strip()]
            fact_lst_lst = [[response.strip()]]

        print(f"Returning facts and token counts for the whole response ...")
        return snippet_lst, fact_lst_lst, all_facts_lst, prompt_tok_cnt, response_tok_cnt
    



    async def veriscore_extractor(self, response, cost_estimate_only=False):
        """
        Given a model output
        - split the response into sentences using spaCy
        - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """
        sentences = self.get_sentence(response)

        all_facts_lst = []
        # keep track of token counts
        prompt_tok_cnt, response_tok_cnt = 0, 0

        # new return values
        snippet_lst = []
        fact_lst_lst = []

        semaphore = asyncio.Semaphore(2)
        async def sem_fact_extractor(snippet, sentence):
            async with semaphore:
                return await self.fact_extractor(snippet, sentence, qa_input=False, cost_estimate_only=cost_estimate_only)

        tasks = []
        for i, sentence in enumerate(sentences):
            if self.model:
                input = response.strip()
                snippet = input.replace(sentence, f"<SOS>{sentence}<EOS>")
            else:
                lead_sent = sentences[0]  # 1st sentence of the para
                context1 = " ".join(sentences[max(0, i - 3):i])
                sentence = f"<SOS>{sentences[i].strip()}<EOS>"
                context2 = " ".join(sentences[i + 1:i + 2])

                # if the para is not long
                if len(sentences) <= 5:
                    snippet = f"{context1.strip()} {sentence.strip()} {context2.strip()}".strip()
                # if the para is long, add lead sentence to context1
                else:
                    snippet = f"{lead_sent.strip()} {context1.strip()} {sentence.strip()} {context2.strip()}".strip()
            snippet_lst.append(snippet)
            tasks.append(sem_fact_extractor(snippet, sentences[i].strip()))

        results = await asyncio.gather(*tasks)

        for facts, prompt_tok_num, response_tok_num in results:
            prompt_tok_cnt += prompt_tok_num
            response_tok_cnt += response_tok_num

            if facts is None:
                fact_lst_lst.append([None])
                continue

            fact_lst = []
            for fact in facts:
                if fact.strip() == "":
                    continue
                elif fact.startswith("Note:"):
                    continue
                elif fact.strip() not in all_facts_lst:
                    all_facts_lst.append(fact.strip())
                fact_lst.append(fact.strip())
            fact_lst_lst.append(fact_lst)

        if len(all_facts_lst) == 0:
            all_facts_lst = [response.strip()]
            fact_lst_lst = [[response.strip()]]

        print(f"Returning facts and token counts for the whole response ...")
        return fact_lst_lst, all_facts_lst, prompt_tok_cnt, response_tok_cnt
    


    async def factscore_extractor(self, response, cost_estimate_only=False):
        """
        Given a model output
        - split the response into sentences using spaCy
        - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """

        generator = AtomicFactGenerator(cache_dir=self.cache_dir, 
                                model_name=self.model_name, 
                                is_bio=False)
        atomic_facts, para_breaks = generator.run(response, cost_estimate=False)
        fact_lst_lst = [item[1] for item in atomic_facts]
        
        # sentences = self.get_sentence(response)

        all_facts_lst = []
        for fact_lst in fact_lst_lst:
            for fact in fact_lst:
                if fact.strip() == "":
                    continue
                if fact.strip().endswith(':'):
                    continue
                # cases where GPT returns its justification
                elif fact.startswith("Note:"):
                    continue
                elif fact.strip() not in all_facts_lst:
                    all_facts_lst.append(fact.strip())

        # keep track of token counts
        prompt_tok_cnt, response_tok_cnt = 0, 0
        
        if len(all_facts_lst) == 0:
            # If no facts are extracted, just use the original response as the only fact
            all_facts_lst = [response.strip()]
            fact_lst_lst = [[response.strip()]]

        print(f"Returning facts from FACTSCORE and token counts (IGNORE THIS token counts) for the whole response ...")
        return fact_lst_lst, all_facts_lst, prompt_tok_cnt, response_tok_cnt




    async def wice_extractor(self, response, cost_estimate_only=False):
        """
        Given a model output
        - split the response into sentences using spaCy
        - snippet = (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """

        def extract_facts(text):
            # Use regex to find all fact lines and remove leading characters
            facts = re.findall(r'(?:-|\d+\.)\s*(.+)', text)
            if len(facts) == 0 and len(text.split('\n')) > 1:
                facts = text.split('\n')
            return [fact.strip() for fact in facts]

        # generator = AtomicFactGenerator(cache_dir=self.cache_dir, 
                                # model_name=self.model_name, 
                                # is_bio=False)
        prompt_text = WICE_PROMPT.format(response.strip())
        response_content, prompt_tok_cnt, response_tok_cnt, response_logprobs = await self.get_model_response.async_get_response(system_message="", 
                                                                                                      prompt_text=prompt_text,
                                                                                                      cost_estimate_only=cost_estimate_only)
        extracted_facts = extract_facts(response_content)

        all_facts_lst = []
        for fact in extracted_facts:
            if fact.strip() not in all_facts_lst:
                all_facts_lst.append(fact.strip())
        fact_lst_lst = [all_facts_lst]

        # keep track of token counts
        prompt_tok_cnt, response_tok_cnt = 0, 0
        
        if len(all_facts_lst) == 0:
            # If no facts are extracted, just use the original response as the only fact
            all_facts_lst = [response.strip()]
            fact_lst_lst = [[response.strip()]]

        print(f"Returning facts from WICE and token counts (IGNORE THIS token counts) for the whole response ...")
        return fact_lst_lst, all_facts_lst, prompt_tok_cnt, response_tok_cnt



    async def specified_number_extractor(self, response, number_of_claims, cost_estimate_only=False):
        def extract_decomposition(decomposed_claim, level):
            extracted_claims = []
            if level == "response":
                pattern = r"```(.*?)```"
                matches = re.finditer(pattern, decomposed_claim, re.DOTALL)
                segments = [match.group(1).strip() for match in matches]
                for segment in segments:
                    if segment.strip() == "":
                        continue
                    extracted_claims.append(segment)
            else:
                decomposition_match = re.search(r'### Sub-claims\n(.*?)(?:\n###|$)', decomposed_claim, re.DOTALL)
                decomposition_match = decomposition_match.group(1).strip() if decomposition_match else None
                extracted_claims = []
                for claim in decomposition_match.split('\n'):
                    if claim.strip() == "":
                        continue
                    if claim.strip().startswith('==='):
                        continue
                    if claim.startswith('-'):
                        claim = claim.strip('-').strip()
                    extracted_claims.append(claim)
            
            if len(extracted_claims) == 0:
                extracted_claims = [decomposed_claim]
            
            return extracted_claims
        
        template = self.specified_number_prompt_template

        prompt_text = template.format(num_sub_claims=number_of_claims, input_text=response.strip())
        response_content, prompt_tok_cnt, response_tok_cnt, response_logprobs = await self.get_model_response.async_get_response(system_message="", 
                                                                                                      prompt_text=prompt_text,
                                                                                                      cost_estimate_only=cost_estimate_only)
        extracted_facts = extract_decomposition(response_content, level=self.input_level)
        fact_lst_lst = [extracted_facts]
        all_facts_lst = extracted_facts


        return fact_lst_lst, all_facts_lst, prompt_tok_cnt, response_tok_cnt
    


    def qa_scanner_extractor(self, question, response, cost_estimate_only=False):
        """
        Given a model output to a question
        - split the response into sentences using spaCy
        - snippet = question (context1 = 0-3 sentence) <SOS>Sent<EOS> (context2 = 0-1 sentence)
        - call fact_extractor on each snippet
        """
        all_facts_lst = []
        # keep track of token counts
        prompt_tok_cnt, response_tok_cnt = 0, 0
        sentences = self.get_sentence(response)

        # new return values
        snippet_lst = []
        fact_lst_lst = []
        for i, sentence in enumerate(sentences):
            if self.model:
                input = f"Questions:\n{question.strip()}\nResponse:\n{response.strip()}"
                snippet = input.replace(sentence, f"<SOS>{sentence}<EOS>")
            else:
                context1 = " ".join(sentences[max(0, i - 3):i])
                sentence = f"<SOS>{sentences[i].strip()}<EOS>"
                context2 = " ".join(sentences[i + 1:i + 2])

                snippet = f"Question: {question.strip()}\nResponse: {context1.strip()} {sentence.strip()} {context2.strip()}".strip()
            # new return value
            snippet_lst.append(snippet)

            # call fact_extractor on each tesnippetxt
            facts, prompt_tok_num, response_tok_num = self.fact_extractor(snippet, sentences[i].strip(), qa_input=True,
                                                                          cost_estimate_only=cost_estimate_only)

            # update token counts
            prompt_tok_cnt += prompt_tok_num
            response_tok_cnt += response_tok_num

            if facts == None:
                fact_lst_lst.append([None])
                continue

            # deduplication
            fact_lst = []
            for fact in facts:
                if fact.strip() == "":
                    continue
                # cases where GPT returns its justification
                elif fact.startswith("Note:"):
                    continue
                elif fact not in all_facts_lst:
                    all_facts_lst.append(fact.strip())
                fact_lst.append(fact.strip())
            fact_lst_lst.append(fact_lst)
        print(f"Returning facts and token counts for the whole response ...")
        return fact_lst_lst, all_facts_lst, prompt_tok_cnt, response_tok_cnt

    def get_sentence(self, text):
        # use spaCy to split the text into sentences
        return [x.text.strip() for x in self.spacy_nlp(text).sents]


    def get_prompt_template(self, qa_input):
        if qa_input:
            prompt_template = open("./prompt/extraction_qa_template.txt", "r").read()
        else:
            prompt_template = open("./prompt/extraction_non_qa_template.txt", "r").read()
        return prompt_template


    async def fact_extractor(self, snippet, sentence, qa_input=False, cost_estimate_only=False):
        if self.model:
            formatted_input = self.alpaca_prompt.format(snippet, "")
            inputs = self.tokenizer(formatted_input, return_tensors="pt").to("cuda")

            outputs = self.model.generate(**inputs, max_new_tokens=1000, use_cache=True)
            output_str = ' '.join(self.tokenizer.batch_decode(outputs))
            clean_output = output_str.split("### Response:")[-1].strip().replace("</s>", "")
            if not clean_output or "No verifiable claim." in clean_output:
                return None, 0, 0
            claims = [x.strip() for x in clean_output.split("\n")]
            return claims, 0, 0
        else:
            prompt_template = self.get_prompt_template(qa_input)
            prompt_text = prompt_template.format(snippet=snippet, sentence=sentence)
            # print("Awating for response...")
            response, prompt_tok_cnt, response_tok_cnt, response_logprobs = await self.get_model_response.async_get_response(self.system_message,
                                                                                                          prompt_text,
                                                                                                          cost_estimate_only)
            # print("response: ", response)
            if not response or "No verifiable claim." in response:
                return None, prompt_tok_cnt, response_tok_cnt
            else:
                claims = [x.strip().replace("- ", "") for x in response.split("\n")]
                claims = [regex.sub(r"^\d+\.?\s", "", x) for x in claims]
                return claims, prompt_tok_cnt, response_tok_cnt


        tasks = [sem_selfcontainlize(claim, ori_claim) for claim in all_claims]
        selfcontained_claims = await asyncio.gather(*tasks)
        return selfcontained_claims
    
    async def self_diagnosis(self, ori_claim, all_claims):
        print(f"\n\nsend to selfdiagnosing claim: {all_claims}")
        prompt_text = get_self_diagnosis_prompt(input_text=ori_claim, sub_claims=all_claims)
        response_content, prompt_tok_cnt, response_tok_cnt, response_logprobs = await self.get_diagnosis_response.async_get_response("", prompt_text)
        print(f"self-diagnosis response: \n{response_content}\n==========\n")
        sections = extract_diagnosis(response_content)
        if sections is None:
            return all_claims, sections
        
        judgment = sections['Judgment']
        improved_decomposition = sections['Refined Decomposition']
        extracted_claims = []
        if improved_decomposition is not None:
            for claim in improved_decomposition.split('\n'):
                if claim.strip() == "":
                    continue
                if claim.strip().startswith('==='):
                    continue
                if 'refined decomposition' in claim.lower():
                    continue
                if claim.startswith('-'):
                    claim = claim.strip('-').strip()
                extracted_claims.append(claim)
        
        if 'good' in judgment.lower():
            return all_claims, sections
        elif 'problematic' in judgment.lower():
            return extracted_claims, sections
        else:
            return all_claims, sections


    async def self_detection(self, ori_claim, all_claims):
        print(f"\n\nsend to selfdetection claim: {all_claims}")
        prompt_text = get_detection_prompt(input_text=ori_claim, sub_claims=all_claims)
        response_content, prompt_tok_cnt, response_tok_cnt, response_logprobs = await self.get_diagnosis_response.async_get_response("", prompt_text)
        print(f"self-detection response: \n{response_content}\n==========\n")
        sections = extract_detection(response_content)
        return sections
