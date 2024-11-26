# get K median and K max
from collections import defaultdict

import numpy as np
from scipy.stats import hmean
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from tqdm import tqdm


def get_stats(domain_model_triplet_dict):
    domain_K_dict = defaultdict(lambda: defaultdict(int))
    for domain, model_triplet_dict in domain_model_triplet_dict.items():
        claim_num_lst = []
        for model_name, triplet_lst in model_triplet_dict.items():
            for triplet in triplet_lst:
                claim_num_lst.append(triplet[1])

        claim_num_lst.sort()
        K_median = claim_num_lst[len(claim_num_lst)//2]
        K_max = claim_num_lst[-1]
        domain_K_dict[domain]["K_median"] = K_median
        domain_K_dict[domain]["K_max"] = K_max
        # print(f"{domain} - {K_median}: {K_max}")

    return domain_K_dict

def get_avg_numbers(domain_model_triplet_dict, domain_K_dict):
    for domain, model_triplet_dict in domain_model_triplet_dict.items():
        K_median = domain_K_dict[domain]["K_median"]
        K_max = domain_K_dict[domain]["K_max"]

        table_content = []
        F1_at_median_lst = []
        for model_name in model_triplet_dict.keys():
            triplet_lst = domain_model_triplet_dict[domain][model_name]

            sent_len_lst = [x[2] for x in triplet_lst]
            sup_lst = [x[0] for x in triplet_lst]
            uns_lst = [x[1] - x[0] for x in triplet_lst]
            prec_lst = [x[0] / x[1] for x in triplet_lst]
            rec_med_lst = [min(x[0] / K_median, 1) for x in triplet_lst]
            rec_max_lst = [min(x[0] / K_max, 1) for x in triplet_lst]

            # get f1@K median and f1@K max
            f1_med_lst = [2 * prec * rec_med / (prec + rec_med) if rec_med > 0 else 0 for prec, rec_med in zip(prec_lst, rec_med_lst)]
            f1_max_lst = [2 * prec * rec_max / (prec + rec_max) if rec_max > 0 else 0 for prec, rec_max in zip(prec_lst, rec_max_lst)]

            # get ave. numbers
            ave_sent = sum(sent_len_lst) / len(sent_len_lst)
            S = sum(sup_lst) / len(sup_lst)
            U = sum(uns_lst) / len(uns_lst)
            P = sum(prec_lst) / len(prec_lst)
            Rec_med = sum(rec_med_lst) / len(rec_med_lst)
            Rec_max = sum(rec_max_lst) / len(rec_max_lst)
            F1_med = sum(f1_med_lst) / len(f1_med_lst)
            F1_max = sum(f1_max_lst) / len(f1_max_lst)

            table_row = [model_name, domain, round(ave_sent, 3), round(S, 3), round(U, 3), round(P, 3), round(Rec_med, 3), round(Rec_max, 3), round(F1_med, 3), round(F1_max, 3)]
            table_content.append(table_row)

            F1_at_median_lst.append(100*round(F1_med, 3))

            print(f"[{domain}-{model_name}] \nF1@k median: {F1_med:.3f}, F1@k max: {F1_max:.3f}")

def get_veriscore(domain_model_triplet_dict):
     domain_K_dict= get_stats(domain_model_triplet_dict)
     get_avg_numbers(domain_model_triplet_dict, domain_K_dict)



def agg_max_hmean(claim_verification_result):
    claim_score_lst = [max(item['claim_nli_score_list']) if len(item['claim_nli_score_list']) > 0 else item['claim_nli_score'] for item in claim_verification_result]
    agg_score = hmean(claim_score_lst)
    return agg_score



def evaluate(veri_results, agg_func=agg_max_hmean):
    for claim_res in tqdm(veri_results):
        claim_res['binary_label'] = 1 if claim_res['annot_label'] == 'supported' else 0
        claim_res['entailment_score'] = agg_func(claim_res['claim_verification_result'])
    
    label_lst = [item['binary_label'] for item in veri_results]
    entailment_score_lst = [item['entailment_score'] for item in veri_results]
    
    # Compute the F1 score, precision, recall, accuracy, AUROC
    sep_claim_lst = [len(item['claim_verification_result']) for item in veri_results]
    avg_sep_claim = sum(sep_claim_lst) / len(sep_claim_lst)

    # Convert entailment scores to binary labels
    binary_preds = [1 if score >= 0.5 else 0 for score in entailment_score_lst]

    # Compute F1 score, precision, recall, accuracy, AUROC
    f1 = f1_score(label_lst, binary_preds)
    precision = precision_score(label_lst, binary_preds)
    recall = recall_score(label_lst, binary_preds)
    accuracy = accuracy_score(label_lst, binary_preds)
    balanced_accuracy = balanced_accuracy_score(label_lst, binary_preds)
    auroc = roc_auc_score(label_lst, entailment_score_lst)

    # All result round to 4 decimal places
    f1 = round(f1, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    accuracy = round(accuracy, 4)
    balanced_accuracy = round(balanced_accuracy, 4)
    auroc = round(auroc, 4)
    avg_sep_claim = round(avg_sep_claim, 4)

    print(f"Accuracy: {accuracy}")
    print(f"Balanced Accuracy: {balanced_accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"AUROC: {auroc}")
    print(f"Claim Count: {avg_sep_claim}")

    metrics_dict = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "claim_count": avg_sep_claim
    }

    return metrics_dict

