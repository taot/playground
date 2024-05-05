use std::collections::HashMap;
use std::path::PathBuf;

use jieba_tokenizer::JiebaEncoding as JiebaEncodingRs;
use jieba_tokenizer::JiebaTokenizer as JiebaTokenizerRs;
use pyo3::prelude::*;
use pyo3::types::PyIterator;

#[pymodule]
fn jieba_tokenizer_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<JiebaEncoding>()?;
    m.add_class::<JiebaTokenizer>()?;
    Ok(())
}

#[pyclass]
#[derive(Debug, Clone)]
struct JiebaEncoding {

    #[pyo3(get)]
    ids: Vec<i64>,

    #[pyo3(get)]
    tokens: Vec<String>,

    #[pyo3(get)]
    special_tokens_mask: Vec<i8>,
}

impl From<JiebaEncodingRs> for JiebaEncoding {
    fn from(encoding: JiebaEncodingRs) -> Self {
        Self {
            ids: encoding.ids,
            tokens: encoding.tokens,
            special_tokens_mask: encoding.special_tokens_mask
        }
    }
}

impl Into<JiebaEncodingRs> for JiebaEncoding {
    fn into(self) -> JiebaEncodingRs {
        JiebaEncodingRs {
            ids: self.ids,
            tokens: self.tokens,
            special_tokens_mask: self.special_tokens_mask
        }
    }
}

#[pyclass]
struct JiebaTokenizer {
    inner: JiebaTokenizerRs
}

#[pymethods]
impl JiebaTokenizer {

    #[new]
    fn new() -> Self {
        let inner = JiebaTokenizerRs::new();
        JiebaTokenizer { inner }
    }

    #[staticmethod]
    fn from_file(file_path: PathBuf) -> Self {
        println!("Creating from file {:?}", file_path);

        let inner = JiebaTokenizerRs::from_file(file_path);
        Self::from_tokenizer_rs(inner)
    }

    fn save(&self, file_path: PathBuf) -> () {
        self.inner.save(file_path);
    }

    fn encode(&self, sequence: &str, add_special_tokens: bool) -> JiebaEncoding {
        self.inner.encode(sequence, add_special_tokens).into()
    }

    fn encode_batch(&self, sequences: Vec<String>, add_special_tokens: bool) -> Vec<JiebaEncoding> {
        sequences.iter().map(|seq| self.encode(seq, add_special_tokens)).collect()
    }

    fn decode(&self, ids: Vec<i64>, skip_special_tokens: bool) -> String {
        self.decode_internal(&ids, skip_special_tokens)
    }

    fn decode_batch(&self, sequences: Vec<Vec<i64>>, skip_special_tokens: bool) -> Vec<String> {
        sequences.iter().map(|ids| self.decode_internal(ids, skip_special_tokens)).collect()
    }

    fn get_vocab(&self) -> HashMap<String, i64> {
        self.inner.get_vocab()
    }

    fn id_to_token(&self, id: i64) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn token_to_id(&self, token: &str) -> Option<i64> {
        self.inner.token_to_id(token)
    }

    fn train_from_iterator(&mut self, iterator: &PyIterator, min_frequency: Option<i64>) -> () {

        let converted_iterator = iterator.map(|it| it.unwrap().str().unwrap().to_str().unwrap());

        self.inner.train_from_iterator(converted_iterator, min_frequency);
    }

    fn normalize(&self, sequence: &str) -> String {
        self.inner.normalize(sequence)
    }

    fn pre_tokenize<'a>(&'a self, sequence: &'a str) -> Vec<&str> {
        self.inner.pre_tokenize(sequence)
    }
}

impl JiebaTokenizer {

    fn from_tokenizer_rs(tokenizer_rs: JiebaTokenizerRs) -> Self {
        JiebaTokenizer { inner: tokenizer_rs }
    }

    fn decode_internal(&self, ids: &Vec<i64>, skip_special_tokens: bool) -> String {
        self.inner.decode(ids, skip_special_tokens)
    }
}
