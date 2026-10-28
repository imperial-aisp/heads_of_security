// Site-wide links and switches.

/** arXiv identifier of the paper. */
export const ARXIV_ID = '2610.37657';

export const LINKS = {
  paper: `https://arxiv.org/abs/${ARXIV_ID}`,
  code: 'https://github.com/imperial-aisp/heads_of_security',
  models: 'https://huggingface.co/collections/imperial-cpg/heads-of-security',
  space: 'https://huggingface.co/spaces/igorshilov/heads-of-security',
  demos: 'https://github.com/imperial-aisp/heads_of_security/tree/main/demos',
  roleConfusion: 'https://openreview.net/forum?id=Lm3DQDyIsI',
};

/** Tutorial notebook on GitHub; the Colab link is derived from it. */
export const NOTEBOOK_URL = 'https://github.com/imperial-aisp/heads_of_security/blob/main/demos/introduction.ipynb';
export const COLAB_URL = NOTEBOOK_URL.replace('https://github.com/', 'https://colab.research.google.com/github/');

/** Show the abstract expanded on page load. */
export const ABSTRACT_OPEN = false;

export const PIP_INSTALL = 'pip install heads-of-security';

export const QUICKSTART = [
  'from heads_of_security import load_gated_model',
  'model, tokenizer = load_gated_model("imperial-cpg/hos-gpt-oss-20b")',
  'enc = tokenizer.apply_chat_template_with_channels(messages, tools=tools)',
  'out = model.generate(',
  '    input_ids=enc["input_ids"],',
  '    attention_mask=enc["attention_mask"],',
  '    token_channels=enc["token_channels"],',
  ')',
].join('\n');

export const BIBTEX = [
  '@article{shilov2026heads,',
  '  title   = {Heads of Security: Learned Attention Gates Improve Prompt Injection Robustness},',
  '  author  = {Shilov, Igor and Yang, Xiaoxue and Panfilov, Alexander and Kortukov, Evgenii and',
  '             Meeus, Matthieu and Romov, Peter and Samek, Wojciech and Geiping, Jonas and',
  '             de Montjoye, Yves-Alexandre},',
  `  journal = {arXiv preprint arXiv:${ARXIV_ID}},`,
  '  year    = {2026},',
  `  url     = {https://arxiv.org/abs/${ARXIV_ID}}`,
  '}',
].join('\n');
