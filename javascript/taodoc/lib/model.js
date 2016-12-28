function ApiDoc(title) {
    this.title = title;
    this.version = '';
    this.basePath = '';
    this.description = '';
    this.interfaces = [];
    this.definitions = [];
}

function Iface(method, path) {
    this.method = method;
    this.path = path;
    this.description = '';
    this.request = undefined;
    this.responses = {};
}

function HttpEntity(headers, body) {
    this.headers = headers;
    this.body = body;
}

function HttpBody(contentType, schema) {
    this.contentType = contentType;
    this.schema = schema;
    this.description = '';
}

function Definition(name) {
    this.name = name;
    
}
