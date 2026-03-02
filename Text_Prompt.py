import torch
import clip
import os


label_text_map = []

with open('text/ntu120_label_map.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        label_text_map.append(line.rstrip().lstrip())



paste_text_map0 = []

with open('text/synonym_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(',')
        paste_text_map0.append(temp_list)
        

paste_text_map1 = []

with open('text/sentence_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split('.')
        while len(temp_list) < 4:
            temp_list.append(" ")
        paste_text_map1.append(temp_list)


paste_text_map2 = []

with open('text/pasta_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        paste_text_map2.append(temp_list)

ucla_paste_text_map0 = []

with open('text/ucla_synonym_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(',')
        ucla_paste_text_map0.append(temp_list)


ucla_paste_text_map1 = []

with open('text/ucla_pasta_openai_t01.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        ucla_paste_text_map1.append(temp_list)




def text_prompt():
    text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                f"The man is {{}}", f"The woman is {{}}"]
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in label_text_map])


    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug,text_dict


def text_prompt_openai_random():
    print("Use text prompt openai synonym random")

    total_list = []
    for pasta_list in paste_text_map0:
        temp_list = []
        for item in pasta_list:
            temp_list.append(clip.tokenize(item))
        total_list.append(temp_list)
    return total_list

def text_prompt_openai_random_bert():
    print("Use text prompt openai synonym random bert")
    
    total_list = []
    for pasta_list in paste_text_map0:
        temp_list = []
        for item in pasta_list:
            temp_list.append(item)
        total_list.append(temp_list)
    return total_list



def text_prompt_openai_pasta_pool_4part():
    print("Use text prompt openai pasta pool")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in paste_text_map2])
        elif ii == 1:
            text_dict[ii] = torch.cat([clip.tokenize((','.join(pasta_list[0:2]))) for pasta_list in paste_text_map2])
        elif ii == 2:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','.join(pasta_list[2:4]))) for pasta_list in paste_text_map2])
        elif ii == 3:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[4])) for pasta_list in paste_text_map2])
        else:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0]+','+','.join(pasta_list[5:]))) for pasta_list in paste_text_map2])


    classes = torch.cat([v for k, v in text_dict.items()])
    
    return classes, num_text_aug, text_dict

def text_prompt_openai_pasta_pool_4part_bert():
    print("Use text prompt openai pasta pool bert")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            input_list = [pasta_list[ii] for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        elif ii == 1:
            input_list = [','.join(pasta_list[0:2]) for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        elif ii == 2:
            input_list = [pasta_list[0] +','.join(pasta_list[2:4]) for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        elif ii == 3:
            input_list = [pasta_list[0] +','+ pasta_list[4] for pasta_list in paste_text_map2]
            text_dict[ii] = input_list
        else:
            input_list = [pasta_list[0]+','+','.join(pasta_list[5:]) for pasta_list in paste_text_map2]
            text_dict[ii] = input_list

    
    return num_text_aug, text_dict



def text_prompt_openai_random_ucla():
    print("Use text prompt openai synonym random UCLA")

    total_list = []
    for pasta_list in ucla_paste_text_map0:
        temp_list = []
        for item in pasta_list:
            temp_list.append(clip.tokenize(item))
        total_list.append(temp_list)
    return total_list


def text_prompt_openai_pasta_pool_4part_ucla():
    print("Use text prompt openai pasta pool ucla")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in ucla_paste_text_map1])
        elif ii == 1:
            text_dict[ii] = torch.cat([clip.tokenize((','.join(pasta_list[0:2]))) for pasta_list in ucla_paste_text_map1])
        elif ii == 2:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','.join(pasta_list[2:4]))) for pasta_list in ucla_paste_text_map1])
        elif ii == 3:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0] +','+ pasta_list[4])) for pasta_list in ucla_paste_text_map1])
        else:
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[0]+','+','.join(pasta_list[5:]))) for pasta_list in ucla_paste_text_map1])



    classes = torch.cat([v for k, v in text_dict.items()])
    
    return classes, num_text_aug, text_dict


def text_prompt_hockey_pasta_pool_4part(data_path='./text/'):
    """
    Load Hockey PASTA text and split by body parts, matching NTU's structure.

    hockey_pasta.txt has 6 semicolon-delimited segments per line:
        "action_name, head_desc; hand_desc; arm_desc; hip_desc; leg_desc; foot_desc"

    These are mapped to 4 body-part groups (matching NTU):
        ii=0: global  — "A hockey player {action_name}"
        ii=1: head    — "{action}, {head_desc}"
        ii=2: hands   — "{action}, {hand_desc}, {arm_desc}"
        ii=3: hips    — "{action}, {hip_desc}"
        ii=4: legs    — "{action}, {leg_desc}, {foot_desc}"

    Returns:
        classes: concatenated tensor of all text embeddings
        num_text_aug: 5 (global + 4 body parts)
        text_dict: {aug_id: tensor[11, 77]}
    """
    # Load clean labels from hockey_label.txt
    label_path = os.path.join(data_path, 'hockey_label.txt')
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]

    # Load hockey_pasta.txt — 6 semicolon-delimited body parts per line
    pasta_path = os.path.join(data_path, 'hockey_pasta.txt')
    with open(pasta_path, 'r') as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]

    # Parse: segment[0] may contain "action, head_desc" — extract pure head_desc
    pasta_parsed = []
    for line in raw_lines:
        parts = line.split(';')
        first = parts[0]
        head_desc = first.split(',', 1)[1].strip() if ',' in first else first.strip()
        pasta_parsed.append([head_desc] + [p.strip() for p in parts[1:]])
    # pasta_parsed[i] = [head, hand, arm, hip, leg, foot]  (6 elements)

    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            # Global: "A hockey player {action_name}"
            text_dict[ii] = torch.cat([
                clip.tokenize(f"A hockey player {label}")
                for label in labels
            ])
        elif ii == 1:
            # Head: "{action}, {head_desc}"
            text_dict[ii] = torch.cat([
                clip.tokenize(
                    f"{labels[i]}, {pasta_parsed[i][0]}",
                    truncate=True
                )
                for i in range(len(labels))
            ])
        elif ii == 2:
            # Hands: "{action}, {hand_desc}, {arm_desc}"
            text_dict[ii] = torch.cat([
                clip.tokenize(
                    f"{labels[i]}, {pasta_parsed[i][1]}, {pasta_parsed[i][2]}",
                    truncate=True
                )
                for i in range(len(labels))
            ])
        elif ii == 3:
            # Hips: "{action}, {hip_desc}"
            text_dict[ii] = torch.cat([
                clip.tokenize(
                    f"{labels[i]}, {pasta_parsed[i][3]}",
                    truncate=True
                )
                for i in range(len(labels))
            ])
        else:  # ii == 4
            # Legs/Feet: "{action}, {leg_desc}, {foot_desc}"
            text_dict[ii] = torch.cat([
                clip.tokenize(
                    f"{labels[i]}, {pasta_parsed[i][4]}, {pasta_parsed[i][5]}",
                    truncate=True
                )
                for i in range(len(labels))
            ])

    classes = torch.cat([v for k, v in text_dict.items()])
    return classes, num_text_aug, text_dict


def text_prompt_hockey_random(data_path='./text/'):
    """
    Load hockey synonym + sentence text and pre-tokenize for random sampling.
    Returns: list of lists of tokenized tensors (one per class).
    """
    # Load synonyms (comma-separated per class)
    synonym_path = os.path.join(data_path, 'hockey_synonym_gemini_t01.txt')
    with open(synonym_path, 'r') as f:
        synonyms = [line.strip().split(',') for line in f.readlines() if line.strip()]

    # Load sentences (one per class)
    sentence_path = os.path.join(data_path, 'hockey_sentence.txt')
    with open(sentence_path, 'r') as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]

    total_list = []
    for i, class_synonyms in enumerate(synonyms):
        temp_list = []
        for item in class_synonyms:
            temp_list.append(clip.tokenize(item.strip()))
        # Add sentence description to the random pool
        if i < len(sentences):
            temp_list.append(clip.tokenize(sentences[i], truncate=True))
        total_list.append(temp_list)
    return total_list
