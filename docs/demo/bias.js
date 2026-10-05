/* Client-side TF-IDF + Linear SVM, a line-for-line port of scikit-learn's
 *   TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2)) -> LinearSVC
 * plus Platt-scaled probabilities. Works in the browser (window.BiasModel) and in Node. */
(function (root) {
  "use strict";

  // sklearn's default token_pattern r"(?u)\b\w\w+\b" == maximal runs of word chars with length >= 2.
  // Python's \w is "alphanumeric or underscore", i.e. \p{L}, \p{N} and "_".
  const WORD_RUN = /[\p{L}\p{N}_]+/gu;
  const cpLength = (s) => { let n = 0; for (const _ of s) n++; return n; };

  function BiasModel(data) {
    const v = data.vectorizer;
    if (v.token_pattern !== "(?u)\\b\\w\\w+\\b" || v.strip_accents || !v.lowercase ||
        v.use_idf === false || v.norm !== "l2" || v.ngram_range[1] > 2) {
      throw new Error("Unsupported vectorizer configuration");
    }
    this.data = data;
    this.stop = new Set(v.stop_words);
    this.index = new Map(data.terms.map((t, i) => [t, i]));
    this.ngram = v.ngram_range;
    this.sublinear = v.sublinear_tf;
  }

  /** Tokens with their character span in the original text (lowercased, stop words kept). */
  BiasModel.prototype.tokenize = function (text) {
    const out = [];
    for (const m of text.matchAll(WORD_RUN)) {
      if (cpLength(m[0]) >= 2) out.push({ w: m[0].toLowerCase(), start: m.index, end: m.index + m[0].length });
    }
    return out;
  };

  BiasModel.prototype.analyze = function (text, opts) {
    opts = opts || {};
    const d = this.data;
    // 1) tokens -> stop-word removal -> n-grams (sklearn removes stop words *before* building n-grams)
    const toks = this.tokenize(text).filter((t) => !this.stop.has(t.w));
    const counts = new Map();   // term index -> {tf, occ: [[firstTok,lastTok], ...]}
    for (let n = this.ngram[0]; n <= this.ngram[1]; n++) {
      for (let i = 0; i + n <= toks.length; i++) {
        let term = toks[i].w;
        for (let k = 1; k < n; k++) term += " " + toks[i + k].w;
        const idx = this.index.get(term);
        if (idx === undefined) continue;
        let e = counts.get(idx);
        if (!e) counts.set(idx, (e = { tf: 0, occ: [] }));
        e.tf++;
        e.occ.push([i, i + n - 1]);
      }
    }
    // 2) tf-idf (+ optional sublinear tf) and L2 norm over the in-vocabulary features
    let sq = 0;
    const feats = [];
    for (const [idx, e] of counts) {
      const tf = this.sublinear ? 1 + Math.log(e.tf) : e.tf;
      const x = tf * d.idf[idx];
      sq += x * x;
      feats.push({ idx, x, occ: e.occ });
    }
    const norm = Math.sqrt(sq);
    // 3) linear decision function: score = w . x + b ; score > 0  =>  classes[1]
    let score = d.intercept;
    const contribs = [];
    if (norm > 0) {
      for (const f of feats) {
        const c = (d.coef[f.idx] * f.x) / norm;
        score += c;
        contribs.push({ term: d.terms[f.idx], contribution: c, occ: f.occ });
      }
    }
    const pSecond = 1 / (1 + Math.exp(-(d.platt.A * score + d.platt.B)));   // P(classes[1])
    const label = score > 0 ? d.classes[1] : d.classes[0];
    const res = {
      label, score,
      probability: label === d.classes[1] ? pSecond : 1 - pSecond,        // confidence in the shown label
      pNeutral: pSecond,
      wordCount: this.tokenize(text).length,
      knownTerms: feats.length,
      empty: norm === 0,
    };
    if (opts.explain) {
      const top = (sign, k) => contribs.filter((c) => c.contribution * sign > 0)
        .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, k);
      res.toward = { [d.classes[0]]: top(-1, opts.top || 8), [d.classes[1]]: top(+1, opts.top || 8) };
      res.spans = this.highlight(text, toks, [].concat(res.toward[d.classes[0]], res.toward[d.classes[1]]), d.classes);
    }
    return res;
  };

  /** Character ranges to highlight: [{start, end, cls}], strongest term wins where terms overlap. */
  BiasModel.prototype.highlight = function (text, toks, chosen, classes) {
    const win = new Array(toks.length).fill(null);   // per token: winning {strength, cls, id}
    chosen.forEach((c, id) => {
      const cls = c.contribution > 0 ? classes[1] : classes[0];
      const strength = Math.abs(c.contribution);
      for (const [a, b] of c.occ) {
        for (let i = a; i <= b; i++) {
          if (!win[i] || strength > win[i].strength) win[i] = { strength, cls, id: id + ":" + a };
        }
      }
    });
    const spans = [];
    for (let i = 0; i < toks.length; i++) {
      if (!win[i]) continue;
      const last = spans[spans.length - 1];
      const prev = i > 0 ? win[i - 1] : null;
      if (last && prev && prev.id === win[i].id && last.cls === win[i].cls && last.endTok === i - 1) {
        last.end = toks[i].end; last.endTok = i;       // bigram: highlight the gap between its two words
      } else {
        spans.push({ start: toks[i].start, end: toks[i].end, cls: win[i].cls, endTok: i });
      }
    }
    return spans.map(({ start, end, cls }) => ({ start, end, cls }));
  };

  BiasModel.load = async function (url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error("Could not load model: " + r.status);
    return new BiasModel(await r.json());
  };

  if (typeof module !== "undefined" && module.exports) module.exports = BiasModel;
  else root.BiasModel = BiasModel;
})(typeof self !== "undefined" ? self : this);
