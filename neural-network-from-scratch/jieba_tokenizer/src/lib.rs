use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

use jieba_rs::Jieba;
use serde_json::{json, Value};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug, Clone)]
pub struct JiebaEncoding {
    pub ids: Vec<i64>,
    pub tokens: Vec<String>,
    pub special_tokens_mask: Vec<i8>,
}

pub struct JiebaTokenizer {
    data: Vec<(String, i64, i64)>,    // list of tuple (token, id, frequency)
    token_to_id_map: HashMap<String, i64>,
    id_to_token_map: HashMap<i64, String>,
    jieba: Jieba,
    special_token_ids: Vec<i64>,
}

pub const UNK: &str = "[UNK]";
pub const PAD: &str = "[PAD]";
pub const SOS: &str = "[SOS]";
pub const EOS: &str = "[EOS]";

pub const SPECIAL_TOKENS: &[&str] = &[UNK, PAD, SOS, EOS];

impl JiebaTokenizer {

    pub fn new() -> Self {
        JiebaTokenizer {
            data: vec![],
            token_to_id_map: HashMap::new(),
            id_to_token_map: HashMap::new(),
            jieba: Jieba::new(),
            special_token_ids: vec![]
        }
    }

    pub fn from_file(file_path: PathBuf) -> Self {
        let mut tokenizer = JiebaTokenizer::new();

        let file = File::open(file_path).unwrap();
        let reader = BufReader::new(file);
        let json_obj: Value = serde_json::from_reader(reader).unwrap();

        tokenizer.data = json_obj.get("data").unwrap().as_array().unwrap().iter().map(|it: &Value| {
            let arr = it.as_array().unwrap();
            (arr[0].as_str().unwrap().to_string(), arr[1].as_i64().unwrap(), arr[2].as_i64().unwrap())
        }).collect::<Vec<_>>();

        tokenizer.reconstruct_from_data();

        tokenizer
    }

    pub fn save(&self, file_path: PathBuf) -> () {
        let json_data = Value::from(self.data.iter().map(|(token, id, freq)| {
            Value::Array(vec![
                Value::from(token.to_string()),
                Value::from(*id),
                Value::from(*freq),
            ])
        }).collect::<Vec<_>>());
        let json_obj = json!({ "data": json_data });

        let file = File::create(file_path).unwrap();
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &json_obj).unwrap()
    }

    pub fn encode(&self, sequence: &str, add_special_tokens: bool) -> JiebaEncoding {
        let sequence = self.normalize(sequence);
        let pre_tokens = self.pre_tokenize(&sequence);
        let unk_id = self.token_to_id(UNK).unwrap();

        let mut ids = vec![];
        let mut tokens = vec![];
        let mut special_tokens_mask = vec![];

        for pre_tok in pre_tokens {
            match self.token_to_id(pre_tok) {
                Some(id) => {
                    ids.push(id);
                    tokens.push(pre_tok.to_string());
                    special_tokens_mask.push(0);
                },
                None => {
                    ids.push(unk_id);
                    tokens.push(UNK.to_string());
                    special_tokens_mask.push(0);
                }
            }
        }

        let encoding = JiebaEncoding { ids, tokens, special_tokens_mask };
        let encoding = self.post_process(encoding, add_special_tokens);

        encoding
    }

    pub fn encode_batch(&self, sequences: Vec<String>, add_special_tokens: bool) -> Vec<JiebaEncoding> {
        sequences.iter().map(|seq| self.encode(seq, add_special_tokens)).collect()
    }

    pub fn decode(&self, ids: &Vec<i64>, skip_special_tokens: bool) -> String {
        let tokens = ids.iter().filter(|id| ! (skip_special_tokens && self.special_token_ids.contains(id)))
            .map(|&id| self.id_to_token(id).unwrap())
            .collect::<Vec<_>>();

        tokens.join("")
    }

    pub fn decode_batch(&self, sequences: &Vec<Vec<i64>>, skip_special_tokens: bool) -> Vec<String> {
        sequences.iter().map(|ids| self.decode(ids, skip_special_tokens)).collect()
    }

    pub fn get_vocab(&self) -> HashMap<String, i64> {
        self.token_to_id_map.clone()
    }

    pub fn id_to_token(&self, id: i64) -> Option<String> {
        self.id_to_token_map.get(&id).map(|s| s.to_string())
    }

    pub fn token_to_id(&self, token: &str) -> Option<i64> {
        self.token_to_id_map.get(token).map(|id| *id)
    }

    pub fn train_from_iterator<'a>(&mut self, iterator: impl Iterator<Item = &'a str>, min_frequency: Option<i64>) -> () {
        let mut counter = HashMap::<String, i64>::new();

        for sequence in iterator {
            let sequence = self.normalize(&sequence);
            let tokens = self.pre_tokenize(&sequence);
            for tok in tokens {
                match counter.get(tok) {
                    Some(v) => counter.insert(tok.to_string(), v + 1),
                    None => counter.insert(tok.to_string(), 1)
                };
            }
        };

        let filtered = match min_frequency {
            Some(min_freq) => counter.drain().filter(|&(_, freq)| freq >= min_freq).collect(),
            None => counter
        };

        let mut vector  = filtered.iter().collect::<Vec<_>>();
        vector.sort_by(|a, b| b.1.cmp(a.1));

        self.data = vec![];
        let mut id = 0;

        for &tok in SPECIAL_TOKENS {
            self.data.push((tok.to_owned(), id, -1));
            id += 1;
        }

        for (tok, freq) in vector {
            self.data.push((tok.to_owned(), id, *freq));
            id += 1;
        }

        self.reconstruct_from_data();
    }

    pub fn normalize(&self, sequence: &str) -> String {
        sequence.nfc().collect::<String>()
    }

    pub fn pre_tokenize<'a>(&'a self, sequence: &'a str) -> Vec<&str> {
        let tokens = self.jieba.cut(sequence, false);
        let tokens = tokens.iter().map(|s| s.trim()).filter(|s| ! s.is_empty()).collect::<Vec<_>>();
        tokens
    }

    pub fn post_process(&self, encoding: JiebaEncoding, add_special_tokens: bool) -> JiebaEncoding {
        if ! add_special_tokens {
            return encoding;
        }

        let sos_id = self.token_to_id(SOS).unwrap();
        let eos_id = self.token_to_id(EOS).unwrap();

        let ids = [vec![sos_id], encoding.ids, vec![eos_id]].concat();
        let tokens = [vec![SOS.to_string()], encoding.tokens, vec![EOS.to_string()]].concat();
        let special_tokens_mask = [vec![1], encoding.special_tokens_mask, vec![1]].concat();

        JiebaEncoding { ids, tokens, special_tokens_mask }
    }

    /*
     * Private methods
     */
    fn reconstruct_from_data(&mut self) -> () {
        self.token_to_id_map = self.data.iter().map(|(token, id, _)| (token.to_string(), *id)).collect::<HashMap<_, _>>();
        self.id_to_token_map = self.data.iter().map(|(token, id, _)| (*id, token.to_string())).collect::<HashMap<_, _>>();
        self.special_token_ids = SPECIAL_TOKENS.iter().map(|t| self.token_to_id(t).unwrap()).collect();
    }
}
