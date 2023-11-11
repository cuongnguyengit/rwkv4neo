while j + 3 < len(input_ids):
    # for j in range(0, len(input_ids), 3):
    k = j + 4
    if input_ids[j: j + 3] == [15960, 27, 222]:
        for k in range(j + 4, min(len(input_ids), j + 4 + block_size)):
            if input_ids[k] == 631:
                break

        if input_ids[k + 1: k + 4] == [11827, 27, 222]:
            # choice_mask +=  [e for e in range(k + 4, min(len(input_ids), j + block_size))]
            for e in range(k + 4, min(len(input_ids), j + block_size)):
                choice_mask.append(e)
                # iid = input_ids[j: e]
                # atm = attention_mask[j: e]
                # n = len(iid)
                # if n < block_size:
                #     iid += [0] * (block_size - len(iid))
                #     atm += [0] * (block_size - len(atm))
                #
                # result['input_ids'] += [iid]
                # result['attention_mask'] += [atm]
            is_qa = True
    elif input_ids[j: j + 3] == [11827, 27, 222]:
        for k in range(j + 4, min(len(input_ids), j + 4 + block_size)):
            if input_ids[k] == 631:
                break
        # choice_mask += [e for e in range(j + 4, k + 1)]
        for e in range(j + 4, k + 1):
            choice_mask.append(e)
            # iid = input_ids[e - block_size: e]
            # atm = attention_mask[e - block_size: e]
            # n = len(iid)
            # if n < block_size:
            #     iid += [0] * (block_size - len(iid))
            #     atm += [0] * (block_size - len(atm))

            # result['input_ids'] += [iid]
            # result['attention_mask'] += [atm]
        is_qa = True
    j += 3