export SERPER_KEY_PRIVATE="SERPER_KEY_PRIVATE"
export OPENAI_API_BASE="https://api.openai.com/v1"
export OPENAI_API_KEY="OPENAI_API_KEY"



python3 src/pipeline_nli.py \
    --data_dir "./data" \
    --input_dir "./input" \
    --input_file "bingchat_random_combination_3000.json" \
    --output_dir "./output/test" \
    --model_name_extraction "gpt-4o" \
    --model_name_verification "gpt-4o-mini" \
    --decompose_method "specified_number" \
    --specified_number_of_claims 8 \
    --label_n 2 \
    --search_res_num 10 \
    --knowledge_base "google" \
    --input_level "response"

# python3 src/pipeline_nli.py \
#     --data_dir "./data" \
#     --input_dir "./input" \
#     --input_file "bingchat_random_combination_3000.json" \
#     --output_dir "./output/test" \
#     --model_name_extraction "gpt-4o" \
#     --model_name_verification "gpt-4o-mini" \
#     --decompose_method "factscore" \
#     --label_n 2 \
#     --search_res_num 10 \
#     --knowledge_base "google" \
#     --input_level "response" \
#     # --use_self_diagnosis # This is for 'reflection', which will reflect & refine based on the original decomposition

# python3 src/pipeline_nli.py \
#     --data_dir "./data" \
#     --input_dir "./input" \
#     --input_file "bingchat_random_combination_3000.json" \
#     --output_dir "./output/test" \
#     --model_name_extraction "gpt-4o" \
#     --model_name_verification "gpt-4o-mini" \
#     --decompose_method "veriscore" \
#     --label_n 2 \
#     --search_res_num 10 \
#     --knowledge_base "google" \
#     --input_level "response"
