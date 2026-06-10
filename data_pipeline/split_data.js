import fs from 'fs';

const inputPath = 'data/slang_dataset.jsonl';
const lines = fs.readFileSync(inputPath, 'utf8').trim().split('\n');

for (let i = lines.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [lines[i], lines[j]] = [lines[j], lines[i]];
}

const total = lines.length;
const trainEnd = Math.floor(total * 0.90);
const valEnd = Math.floor(total * 0.95);

const splits = {
  'data/splits/train.jsonl': lines.slice(0, trainEnd),
  'data/splits/val.jsonl': lines.slice(trainEnd, valEnd),
  'data/splits/test.jsonl': lines.slice(valEnd),
};

fs.mkdirSync('data/splits', { recursive: true });
for (const [filePath, data] of Object.entries(splits)) {
  fs.writeFileSync(filePath, data.join('\n') + '\n');
  console.log(`${filePath}: ${data.length} examples`);
}
console.log(`\nTotal: ${total} examples split into train/val/test`);