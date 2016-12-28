var ejs = require('ejs');
var yaml = require('js-yaml');
var fs = require('fs');

var doc = yaml.safeLoad(fs.readFileSync('exp/api-example.yaml', 'utf8'));
console.log(doc.title);
ejs.renderFile('./exp/template.ejs', doc, {}, function(err, str){
    if (err) {
        console.error(err);
        return;
    }
    fs.writeFile('./exp/output.html', str, function(err) {
        if (err) {
            console.error(err);
            return;
        }
        console.log('saved');
    });
});
