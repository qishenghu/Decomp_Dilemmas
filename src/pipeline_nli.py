"""
File: pipeline_nli.py

Description:
This file is an modified version based on the amazing VeriScore repository.
VeriScore Repo: https://github.com/Yixiao-Song/VeriScore
"""



import argparse
import asyncio
import json
import logging
import os
from collections import defaultdict
from pprint import pprint

import spacy

from claim_extractor import ClaimExtractor
from claim_verifier import ClaimVerifier
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from search_API import SearchAPI
from search_API_asy import AsyncSearchAPI
from third_party.specified_number_claims import (
    CLAIM_LEVEL_DECOMPOSE_NUM_PROMPT_TEMPLATE,
    RESPONSE_LEVEL_DECOMPOSE_NUM_PROMPT_TEMPLATE)

from tqdm import tqdm
from utils import evaluate

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

abstain_responses = ["I'm sorry, I cannot fulfill that request.",
                     "I'm sorry, I can't fulfill that request.",
                     "I'm sorry, but I cannot fulfill that request.",
                     "I'm sorry, but I can't fulfill that request.",
                     "Sorry, but I can't fulfill that request.",
                     "Sorry, I can't do that."]

class Pipeline(object):
    def __init__(self,
                 decompose_method='veriscore',
                 model_name_extraction='gpt-4o-mini',
                 model_name_verification='gpt-4o-mini',
                 use_external_extraction_model=False,
                 use_external_verification_model=False,
                 use_nli_verification=False,
                 knowledge_base="google", # google or wikipedia
                 knowledge_collection_name="wice_test",
                 data_dir='./data',
                 cache_dir='./data/cache',
                 output_dir='./data_cache',
                 knowledge_base_dir=None,
                 label_n=2,
                 search_res_num=10,
                 input_level="claim",
                 use_self_diagnosis=False,
                 model_name_diagnosis=None,
                 specified_number_of_claims=None,
                 use_self_detection=False
                ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.decompose_method = decompose_method
        self.knowledge_base = knowledge_base
        self.knowledge_collection_name = knowledge_collection_name
        self.vector_store = None
        self.embedding_func = None
        self.use_nli_verification = use_nli_verification
        self.input_level = input_level
        self.use_self_diagnosis = use_self_diagnosis
        self.model_name_diagnosis = model_name_diagnosis
        self.specified_number_of_claims = specified_number_of_claims
        self.use_self_detection = use_self_detection

        if self.knowledge_base == 'wikipedia':
            self.knowledge_base_dir = knowledge_base_dir
            model_kwargs = {'trust_remote_code': True, }
            self.embedding_func = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_400M_v5", model_kwargs=model_kwargs)
            self.vector_store = Chroma(
                collection_name=self.knowledge_collection_name,
                embedding_function=self.embedding_func,
                persist_directory=self.knowledge_base_dir,
            )


        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.spacy_nlp = spacy.load('en_core_web_sm')

        self.system_message_extraction = "You are a helpful assistant who can extract verifiable atomic claims from a piece of text. Each atomic fact should be verifiable against reliable external world knowledge (e.g., via Wikipedia)"
        print(f"cache_dir: {self.cache_dir}")
        self.claim_extractor = ClaimExtractor(model_name_extraction, cache_dir=self.cache_dir,
                                              use_external_model=use_external_extraction_model, input_level=self.input_level,
                                              model_name_diagnosis=self.model_name_diagnosis)
        if self.decompose_method == 'specified_number':
            if self.input_level == "response":
                self.claim_extractor.specified_number_prompt_template = RESPONSE_LEVEL_DECOMPOSE_NUM_PROMPT_TEMPLATE
            else:
                self.claim_extractor.specified_number_prompt_template = CLAIM_LEVEL_DECOMPOSE_NUM_PROMPT_TEMPLATE
        self.fetch_search = SearchAPI()
        self.fetch_search_asy = AsyncSearchAPI(max_concurrent_requests=10)

        demon_dir = os.path.join(self.data_dir, 'demos')
        self.model_name_verification = model_name_verification
        self.model_name_extraction = model_name_extraction
        self.claim_verifier = ClaimVerifier(model_name=model_name_verification, label_n=label_n,
                                            cache_dir=self.cache_dir, demon_dir=demon_dir,
                                            use_external_model=use_external_verification_model,
                                            use_nli_verification=use_nli_verification
                                        )
        self.label_n = label_n
        self.search_res_num = search_res_num

    async def run(self, data, input_file_name):
        try:
            # Run to get classification prediction
            
            ### extract claims ###
            output_file = f"claims_{input_file_name}.jsonl"
            output_path = os.path.join(self.output_dir, output_file)
            logging.info(f"Running pipeline for {self.output_dir}")


            extracted_claims = []
            with open(output_path, "w") as f:
                for dict_item in tqdm(data):
                    ori_claim = dict_item["claim"].strip()
                    prompt_source = dict_item["source"]
                    annot_label = dict_item.get("label", None)

                    # skip abstained responses
                    if ori_claim.strip() in abstain_responses:
                        output_dict = {
                                    "question": "",
                                    "claim": ori_claim,
                                    "abstained": True,
                                    "source": prompt_source,
                                    }
                        f.write(json.dumps(output_dict) + "\n")
                        continue

                    question = ''
                    if self.decompose_method == 'veriscore':
                        print("Using [veriscore] decomposition method")
                        if 'llama' in self.model_name_extraction.lower():
                            self.claim_extractor.get_model_response.stop_tokens = ["<|eot_id|>", "Text:"]
                        claim_list, all_claims, prompt_tok_cnt, response_tok_cnt = await self.async_veriscore_extractor(ori_claim)
                        # print(f"claim_list: {claim_list}")
                    elif self.decompose_method == 'factscore':
                        print("Using [factscore] decomposition method")
                        if 'llama' in self.model_name_extraction.lower():
                            self.claim_extractor.get_model_response.stop_tokens = ["<|eot_id|>", "Please breakdown the following sentence"]
                        claim_list, all_claims, prompt_tok_cnt, response_tok_cnt = await self.claim_extractor.factscore_extractor(
                            ori_claim)
                    elif self.decompose_method == 'original':
                        print("Using [original] decomposition method")
                        claim_list, all_claims, prompt_tok_cnt, response_tok_cnt = [[ori_claim]], [ori_claim], 0, 0
                    elif self.decompose_method == 'wice':
                        print("Using [wice] decomposition method")
                        self.claim_extractor.get_model_response.stop_tokens = ["\n\nSentence:"]
                        if 'llama' in self.model_name_extraction.lower():
                            self.claim_extractor.get_model_response.stop_tokens = ["\n\nSentence:", "<|eot_id|>"]
                        claim_list, all_claims, prompt_tok_cnt, response_tok_cnt = await self.claim_extractor.wice_extractor(ori_claim)
                        all_claims_str = "\n* ".join(all_claims)
                        print(f"Extracted claims: {all_claims_str}")
                    elif 'specified_number' in self.decompose_method.lower():
                        print(f"Using [{self.decompose_method}-{self.specified_number_of_claims}] decomposition method")
                        self.claim_extractor.get_model_response.stop_tokens = ["\nInput: ", "\nClaim: "]
                        if 'llama' in self.model_name_extraction.lower():
                            self.claim_extractor.get_model_response.stop_tokens = ["\nInput: ", "\nClaim: ", "<|eot_id|>"]
                        claim_list, all_claims, prompt_tok_cnt, response_tok_cnt = await self.claim_extractor.specified_number_extractor(ori_claim, number_of_claims=self.specified_number_of_claims)
                        print("[specified_number] - Extracted claims: ")
                        for i, claim in enumerate(all_claims):
                            print(f"[Claim-No.{i+1}]:\n{claim}\n")
                    else:
                        raise ValueError(f"Unknown decompose method: {self.decompose_method}")
                    
                    processed_claims = []
                    diagnosis_sections = None
                    error_detection_sections = None
                    if self.use_self_diagnosis:
                        processed_claims, diagnosis_sections = await self.claim_extractor.self_diagnosis(ori_claim, all_claims)
                        pprint(f"diagnosis_sections: \n{diagnosis_sections}")
                    else:
                        error_detection_sections = await self.claim_extractor.self_detection(ori_claim, all_claims)
                        pprint(f"error_detection_sections: \n{error_detection_sections}")

                    
                    # write output
                    output_dict = {"question": question.strip(),
                                "prompt_source": prompt_source,
                                "ori_claim": ori_claim,
                                "annot_label": annot_label,
                                "prompt_tok_cnt": prompt_tok_cnt,
                                "response_tok_cnt": response_tok_cnt,
                                "model_name_extraction": self.model_name_extraction,
                                "model_name_verification": self.model_name_verification,
                                "decompose_method": self.decompose_method,
                                "search_res_num": self.search_res_num,
                                "abstained": False,  # "abstained": False, "abstained": True
                                "claim_list": claim_list,
                                "all_claims": all_claims,
                                "use_self_diagnosis": self.use_self_diagnosis,
                                "processed_claims": processed_claims,
                                "diagnosis_sections": diagnosis_sections,
                                "error_detection_sections": error_detection_sections,
                                }
                    f.write(json.dumps(output_dict) + "\n")
                    extracted_claims.append(output_dict)

            print(f"claim extraction is done! saved to {output_path}")
            logging.info(f"claim extraction is done! saved to {output_path}")
            if self.use_self_detection:
                import sys
                print(f"self-detection is done! saved to {output_path}")
                print("Stop here.")
                sys.exit(0)

            output_file = f"evidence_{input_file_name}.jsonl"
            if self.knowledge_base == "wikipedia":
                output_file = f"evidence_{input_file_name}_wikipedia.jsonl"
            output_path = os.path.join(self.output_dir, output_file)
            searched_evidence_dict = []
            with open(output_path, "w") as f:
                for dict_item in tqdm(extracted_claims):
                    if dict_item['abstained']:
                        f.write(json.dumps(dict_item) + "\n")
                        searched_evidence_dict.append(dict_item)
                        continue

                    claim_lst = dict_item["all_claims"]
                    if self.use_self_diagnosis:
                        claim_lst = dict_item["processed_claims"]
                    if claim_lst == ["No verifiable claim."]:
                        dict_item["claim_search_results"] = []
                        f.write(json.dumps(dict_item) + "\n")
                        searched_evidence_dict.append(dict_item)
                        continue
                    # claim_snippets = self.fetch_search.get_snippets(claim_lst)
                    claim_snippets = await self.async_get_snippets(claim_lst)
                    
                    dict_item["claim_search_results"] = claim_snippets
                    searched_evidence_dict.append(dict_item)
                    f.write(json.dumps(dict_item) + "\n")
                    f.flush()
            print(f"evidence searching is done! saved to {output_path}")
            logging.info(f"evidence searching is done! saved to {output_path}")
            output_dir = os.path.join(args.output_dir, 'model_output')
            os.makedirs(output_dir, exist_ok=True)
            output_file = f'verification_{input_file_name}_{self.label_n}.jsonl'
            output_path = os.path.join(output_dir, output_file)

            total_prompt_tok_cnt = 0
            total_resp_tok_cnt = 0

            with open(output_path, "w") as f:
                for dict_item in tqdm(searched_evidence_dict):
                    prompt_source = dict_item['prompt_source']
                    claim_search_results = dict_item["claim_search_results"]

                    if dict_item['abstained']:
                        f.write(json.dumps(dict_item) + "\n")
                        continue

                    claim_verify_res_dict, prompt_tok_cnt, response_tok_cnt = await self.async_verifying_claim(claim_search_results, search_res_num=args.search_res_num)

                    
                    dict_item["claim_verification_result"] = claim_verify_res_dict

                    f.write(json.dumps(dict_item) + "\n")

                    total_prompt_tok_cnt += prompt_tok_cnt
                    total_resp_tok_cnt += response_tok_cnt

                    ## for VeriScore calculation
                    triplet = [0, 0, 0]
                    triplet[1] = len(dict_item['all_claims'])
                    triplet[2] = len(dict_item['claim_list'])

            print(f"claim verification is done! saved to {output_path}")
            logging.info(f"claim verification is done! saved to {output_path}")
            print(f"Total cost: {total_prompt_tok_cnt * 10 / 1e6 + total_resp_tok_cnt * 30 / 1e6}")
            logging.info(f"Total cost: {total_prompt_tok_cnt * 10 / 1e6 + total_resp_tok_cnt * 30 / 1e6}")

            eval_data = [json.loads(x) for x in open(output_path, 'r').readlines()]
            metrics_dict = evaluate(eval_data)
            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics_dict, f)

        except Exception as e:
            logger.error(f"Error in run: {e}")
            raise e
    
    async def async_get_snippets(self, claim_lst):
        if self.knowledge_base == "google":
            return await self.fetch_search_asy.get_snippets(claim_lst)
        elif self.knowledge_base == "wikipedia":
            text_claim_snippets_dict = {}
            for claim in claim_lst:
                retr_docs = self.vector_store.similarity_search(claim, k=self.search_res_num)
                text_claim_snippets_dict[claim] = [
                    {'snippet': doc.page_content, 'metadata': doc.metadata, 'knowledge_source': 'wikipedia'} for doc in retr_docs
                ]
            return text_claim_snippets_dict
        else:
            raise ValueError(f"Unknown knowledge base: {self.knowledge_base}")
        

    async def async_verifying_claim(self, claim_search_results, search_res_num=5):
        return await self.claim_verifier.async_verifying_claim(claim_search_results, search_res_num=search_res_num)
    
    async def async_veriscore_extractor(self, response):
        return await self.claim_extractor.veriscore_extractor(response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--input_dir", type=str, default='./input')
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default='./data')
    parser.add_argument("--cache_dir", type=str, default='./data/cache')
    parser.add_argument("--model_name_extraction", type=str, default="gpt-4o-mini")
    parser.add_argument("--model_name_verification", type=str, default="")
    parser.add_argument("--model_name_diagnosis", type=str, default="gpt-4o-mini")
    parser.add_argument("--label_n", type=int, default=2, choices=[2, 3])
    parser.add_argument("--search_res_num", type=int, default=10)
    parser.add_argument("--decompose_method", type=str, default='veriscore', choices=['original', 'veriscore', 'factscore', 'wice', 'specified_number'])
    parser.add_argument("--knowledge_base", type=str, default="google", choices=["google", "wikipedia"])
    parser.add_argument("--knowledge_collection_name", type=str, default="wice_test")
    parser.add_argument("--knowledge_base_dir", type=str, default="./wice_test_chroma_db")
    parser.add_argument("--use_nli_verification", action="store_true")
    parser.add_argument("--input_level", type=str, default="response", choices=["claim", "response"])
    parser.add_argument("--use_self_diagnosis", action="store_true")
    parser.add_argument("--specified_number_of_claims", type=int, default=-1)
    parser.add_argument("--use_self_detection", action="store_true")
    parser.add_argument("--data_source", type=str, default="")
    args = parser.parse_args()

    if args.knowledge_base == "wikipedia":
        assert os.path.exists(args.knowledge_base_dir), "Knowledge base directory does not exist"
    
    if args.use_self_diagnosis:
        if args.model_name_diagnosis == "":
            args.model_name_diagnosis = args.model_name_extraction
    
    if args.decompose_method == 'specified_number':
        assert args.specified_number_of_claims != -1, "specified_number_of_claims must be provided"
    

    print("args: ", args)
    pipe = Pipeline(model_name_extraction=args.model_name_extraction,
                    model_name_verification=args.model_name_verification,
                    decompose_method=args.decompose_method,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    cache_dir=args.cache_dir,
                    label_n=args.label_n,
                    search_res_num=args.search_res_num,
                    use_nli_verification=args.use_nli_verification,
                    knowledge_base=args.knowledge_base,
                    knowledge_collection_name=args.knowledge_collection_name,
                    knowledge_base_dir=args.knowledge_base_dir,
                    input_level=args.input_level,
                    use_self_diagnosis=args.use_self_diagnosis,
                    model_name_diagnosis=args.model_name_diagnosis,
                    specified_number_of_claims=args.specified_number_of_claims,
                    use_self_detection=args.use_self_detection
                )

    input_file_name = "".join(args.input_file.split('.')[:-1])
    input_path = os.path.join(args.input_dir, args.input_file)
    if 'jsonl' in args.input_file:
        with open(input_path, "r") as f:
            data = [json.loads(x) for x in f.readlines() if x.strip()]
    else:
        data = json.load(open(input_path, 'r'))
    print("data loaded.")

    if args.data_source != "":
        split_data = [x for x in data if x['source'] == args.data_source]
        data = split_data

    asyncio.run(pipe.run(data, input_file_name))