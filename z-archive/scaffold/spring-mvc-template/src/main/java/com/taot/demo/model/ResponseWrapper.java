package com.taot.demo.model;

public class ResponseWrapper {

    private String code;
    private Object data;
    private String message;

    public ResponseWrapper(String code, String message) {
        this.code = code;
        this.data = null;
        this.message = message;
    }

    public ResponseWrapper(Object data) {
        this.data = data;
        this.code = "SUCCESS";
        this.message = "";
    }

    public String getCode() {
        return code;
    }

    public Object getData() {
        return data;
    }

    public String getMessage() {
        return message;
    }

    @Override
    public String toString() {
        return "ResponseWrapper{" +
                "code='" + code + '\'' +
                ", data=" + data +
                ", message='" + message + '\'' +
                '}';
    }
}
