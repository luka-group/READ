import json

with open("fever.train.jsonl", "r") as f:
    preprocessed_lines = []
    for line in f.read().splitlines():
        data = json.loads(line)
        data["label"] = data["gold_label"]
        del data["gold_label"]
        del data["id"]
        del data["weight"]
        preprocessed_lines.append(json.dumps(data))

with open("fever_train_preprocessed.json", "w") as f:
    f.write("\n".join(preprocessed_lines))

with open("fever.dev.jsonl", "r") as f:
    preprocessed_lines = []
    for line in f.read().splitlines():
        data = json.loads(line)
        data["label"] = data["gold_label"]
        del data["gold_label"]
        del data["id"]
        preprocessed_lines.append(json.dumps(data))

with open("fever_dev_preprocessed.json", "w") as f:
    f.write("\n".join(preprocessed_lines))

with open("fever_symmetric_generated.jsonl", "r") as f:
    preprocessed_lines = []
    for line in f.read().splitlines():
        data = json.loads(line)
        data["evidence"] = data["evidence_sentence"]
        del data["evidence_sentence"]
        del data["id"]
        preprocessed_lines.append(json.dumps(data))

with open("fever_symmetric_preprocessed.json", "w") as f:
    f.write("\n".join(preprocessed_lines))