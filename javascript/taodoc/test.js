fs   = require('fs');
yaml = require('js-yaml');


// Get document, or throw exception on error
try {
    var doc = yaml.safeLoad(fs.readFileSync('tests/api-example.yaml', 'utf8'));
    console.log(JSON.stringify(doc, undefined, 4));
} catch (e) {
    console.log(e);
}
