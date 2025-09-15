input_path = "/home/kuangph/CS336-Assignment1/tests/fixtures/gpt2_merges.txt"
output_path = "/home/kuangph/CS336-Assignment1/data/gpt2_merges.txt"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        fout.write(line.replace("Ġ", " "))