// Helper for parity_test.py: classify a JSON array of texts with the browser model, in Node.
const fs = require("fs");
const path = require("path");
const BiasModel = require(path.join(__dirname, "../docs/demo/bias.js"));
const model = new BiasModel(JSON.parse(fs.readFileSync(path.join(__dirname, "../docs/demo/model.json"), "utf8")));
const texts = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = { label: [], score: [], pNeutral: [] };
for (const t of texts) {
  const r = model.analyze(t);
  out.label.push(r.label); out.score.push(r.score); out.pNeutral.push(r.pNeutral);
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
