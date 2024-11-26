CLAIM_LEVEL_DECOMPOSE_NUM_PROMPT_TEMPLATE = """
Please help me decompose an input text into a set of exactly {num_sub_claims} sub-claims while perserving the original meaning. 
Each sub-claim is required to be *self-contained*, meaning it is completely interpretable without the original text and other sub-claims.

Please directly provide the decomposed set of sub-claims in the following format:

### Sub-claims
- sub-claim #1
- sub-claim #2
...
- sub-claim #{num_sub_claims}

Now, please help me decompose the following input text:

### Input Text
{input_text}
"""


RESPONSE_LEVEL_DECOMPOSE_NUM_PROMPT_TEMPLATE = """
Fact-checking involves accessing the veracity of a given text, which could be a statement, claim, paragraph, or a generated response from a large language model.

For more complex fact-checking tasks, the input text is often broken down into a set of manageable, verifiable and self-contained claims, a process called 'Input Decomposition'. Each claim is required to be *self-contained*, meaning it is completely interpretable without the original text and other claims. Each claim is verified independently, and the results are combined to assess the overall veracity of the original input.

Your task is to decompose an input text into exactly {num_sub_claims} claims while preserving the original meaning. Each claim should be *self-contained*, meaning it is completely interpretable without the original text and other claims.

Please directly provide the decomposed set of claims in the following format, each claim should be enclosed with triple backticks:

### Claims
```
claim #1
```

```
claim #2
```
...
```
claim #{num_sub_claims}
```

Now, please help me decompose the following input text:

### Input Text
```
{input_text}
```
"""
