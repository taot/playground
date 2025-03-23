from transformers import AutoTokenizer

from paths import PROJ_DIR

tokenizer = AutoTokenizer.from_pretrained(PROJ_DIR / "model/minimind_tokenizer")

print(f"{tokenizer.bos_token=}")
print(f"{tokenizer.eos_token=}")

# text = "鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。"

text = "鉴别"

encoding = tokenizer(text=text, max_length=10, padding="max_length", truncation=True, return_tensors="pt")

print(f"{encoding=}")

print(f"{tokenizer.decode(encoding.input_ids.squeeze())=}")

print(f"{tokenizer.decode(token_ids=[1415])=}")
