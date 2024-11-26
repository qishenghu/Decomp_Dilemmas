"""
File: search_API_asy.py

Description:
This file is an modified version based on the amazing VeriScore repository.
VeriScore Repo: https://github.com/Yixiao-Song/VeriScore
"""

import asyncio
import json
import os
import pdb
from ast import literal_eval

import aiohttp
import requests
from tqdm import tqdm


class AsyncSearchAPI():
    def __init__(self, max_concurrent_requests=5):
        # invariant variables
        self.serper_key = os.getenv("SERPER_KEY_PRIVATE")
        self.url = "https://google.serper.dev/search"
        self.headers = {'X-API-KEY': self.serper_key,
                        'Content-Type': 'application/json'}
        # cache related
        self.cache_file = "data/cache/search_cache.json"
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.save_interval = 500
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)


    async def get_snippets(self, claim_lst):
        text_claim_snippets_dict = {}
        tasks = [self.get_search_res(query) for query in claim_lst]
        results = await asyncio.gather(*tasks)
        
        for query, search_result in zip(claim_lst, results):
            if "statusCode" in search_result:
                print(search_result['message'])
                exit()
            organic_res = search_result.get("organic", [])
            search_res_lst = [{"title": item.get("title", ""),
                               "snippet": item.get("snippet", ""),
                               "link": item.get("link", ""),
                               "knowledge_source": "google"} for item in organic_res]
            text_claim_snippets_dict[query] = search_res_lst
        return text_claim_snippets_dict

    async def get_search_res(self, query):
        # check if prompt is in cache; if so, return from cache
        cache_key = query.strip()
        if cache_key in self.cache_dict:
            cached_res = self.cache_dict[cache_key]
            if 'statusCode' in cached_res and cached_res['statusCode'] == 400:
                print(f"Cache hit, but status code is 400, re-fetching...")
                pass
            else:
                return cached_res

        truncated_query = query
        if len(query) > 2048:
            truncated_query = query[:2047]
            print(f"Query is too long({len(query)}), truncated to {len(truncated_query)}.")
        assert len(truncated_query) <= 2048, "Query is too long"
        payload = json.dumps({"q": truncated_query})
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=self.headers, data=payload) as response:
                response_text = await response.text()
                response_json = literal_eval(response_text)

        # update cache
        async with self.lock:
            self.cache_dict[query.strip()] = response_json
            self.add_n += 1

            # save cache every save_interval times
            if self.add_n % self.save_interval == 0:
                await self.save_cache()

        return response_json

    async def save_cache(self):
        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        cache = self.load_cache().items()
        for k, v in cache:
            self.cache_dict[k] = v
        print(f"Saving search cache ...")
        with open(self.cache_file, "w") as f:
            json.dump(self.cache_dict, f, indent=4)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                # load a json file
                cache = json.load(f)
                print(f"Loading cache ...")
        else:
            cache = {}
        return cache

