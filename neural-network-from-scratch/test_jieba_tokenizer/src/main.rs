use std::path::PathBuf;
use jieba_tokenizer;

fn main() {
    println!("Hello, world!");
    let path = PathBuf::from("/home/taot/data/ml_data/my_backup/transformer_from_scratch/tokenizer_zh.json");
    let tokenizer = jieba_tokenizer::JiebaTokenizer::from_file(path);
    let encoding = tokenizer.encode("我们中出了一个叛徒，土豆面包1984", true);
    println!("{:?}", encoding);
}
