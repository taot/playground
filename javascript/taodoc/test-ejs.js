var ejs = require('ejs');
var yaml = require('js-yaml');
var fs = require('fs');

var doc = yaml.safeLoad(fs.readFileSync('tests/api-example.yaml', 'utf8'));
console.log(doc.title);
ejs.renderFile('./tests/template.ejs', doc, {}, function(err, str){
    if (err) {
        console.error(err);
        return;
    }
    fs.writeFile('./tests/output.html', str, function(err) {
        if (err) {
            console.error(err);
            return;
        }
        console.log('saved');
    });
});
