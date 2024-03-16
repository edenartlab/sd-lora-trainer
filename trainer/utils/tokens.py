from collections import OrderedDict

def obtain_inserting_list_tokens(token_list):
    token_dict = OrderedDict({})
    all_token_lists = []
    running_tok_cnt = 0
    for token in token_list:
        token_name, n_tok = token.split(":")
        n_tok = int(n_tok)
        special_tokens = [f"<s{i + running_tok_cnt}>" for i in range(n_tok)]
        token_dict[token_name] = "".join(special_tokens)
        all_token_lists.extend(special_tokens)
        running_tok_cnt += n_tok

    return all_token_lists, token_dict